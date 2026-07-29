from django.conf import settings
from django.http import StreamingHttpResponse, JsonResponse
import cv2
from django.views.decorators.http import require_http_methods
from .yolo import detect_objects
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from .models import *
from django.utils import timezone
from django.contrib.auth import get_user_model
import re
from ultralytics import YOLO
import base64
from django.core.files.storage import FileSystemStorage
import os
from django.core.mail import send_mail

User = get_user_model()

def home(request):
    return render(request, 'index.html')


def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        subject = request.POST.get('subject')
        message = request.POST.get('message')

        # Construct the message content
        full_message = f"Name: {name}\nEmail: {email}\nPhone: {phone}\n\nMessage:\n{message}"

        try:
            # Send the email
            send_mail(
                subject,
                full_message,
                settings.EMAIL_HOST_USER,  # From email
                # Replace with your email or recipient list
                [''],
                fail_silently=False,
            )
            messages.success(
                request, "Your message has been sent successfully!")
        except Exception as e:
            messages.error(request, f"Failed to send message: {str(e)}")

        # Redirect to the same page or another page after submission
        return redirect('contact')

    return render(request, 'contact.html')


def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']

        if len(username) < 6:
            messages.error(
                request, "Username must be at least 6 characters long")
            return redirect('login')

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            messages.error(request, "Invalid email address")
            return redirect('login')

        if len(password1) < 8:
            messages.error(
                request, "Password must be at least 8 characters long")
            return redirect('login')

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken")
            return redirect('login')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already taken")
            return redirect('login')

        user = User.objects.create_user(
            username=username, email=email, password=password1)
        user.save()
        messages.success(request, "User successfully added to database")
        return redirect('login')

    return render(request, 'login.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username=username, password=password)

        if user is not None:
            login(request, user)
            # Redirect to the homepage after a successful login
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('home')


@login_required(login_url='login')
def about(request):
    return render(request, 'about.html')


# Define allowed file types
ALLOWED_FILE_TYPES = ['jpg', 'jpeg', 'png']

CLASS_COLORS = {
    'person': (255, 0, 0),  # Red
    'car': (0, 255, 0),  # Green
    'dog': (0, 0, 255),  # Blue
    'bicycle': (255, 255, 0),  # Cyan
    'cat': (255, 0, 255),  # Magenta
    'truck': (0, 255, 255),  # Yellow
    # Add more class-color mappings as needed
}

def detect(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Check if file extension is valid
        if file_extension not in ALLOWED_FILE_TYPES:
            messages.error(
                request, "Invalid file type! Please upload a .jpg, .jpeg, or .png file.")
            return render(request, 'detect.html')

        # Save the uploaded file
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        # Load YOLOv8 model (replace with the path to your .pt model)
        model = YOLO('yolov8n.pt')

        # Load image
        image = cv2.imread(file_path)

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Perform inference with YOLOv8
        results = model(image)

        # Extracting information from results
        detected_classes = []

        # Draw bounding box and label on image
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

            for i, box in enumerate(boxes):
                confidence = confidences[i]
                class_id = class_ids[i]

                # Only process if confidence is higher than 0.5
                if confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box)
                    label = model.names[class_id]
                    detected_classes.append(label)

                    # Choose color for each class
                    color = CLASS_COLORS.get(label, (0, 255, 255))

                    # Calculate font size and thickness based on image size
                    # Adjust for higher resolutions
                    font_scale = max(img_width, img_height) / 1000
                    # Ensure visibility
                    font_thickness = max(1, int(img_width / 500))

                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # Draw label with confidence
                    text = f"{label} ({confidence:.2f})"
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                                  (text_x + text_size[0] + 5, text_y + 5), color, -1)
                    cv2.putText(image, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Convert image to base64 for displaying on the webpage
        _, buffer = cv2.imencode('.jpg', image)
        image_as_text = base64.b64encode(buffer).decode('utf-8')

        # Delete the uploaded file after processing
        os.remove(file_path)

        # Pass detected_classes and image to the template
        return render(request, 'detect.html', {'detected_classes': detected_classes, 'image_as_text': image_as_text})

    return render(request, 'detect.html')


# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Path to your YOLOv8 model

# Global variables for camera and speed
camera = None
speed = 100  # Start with a speed of 100 km/h (max speed)


def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Start the camera


def stop_camera():
    global camera
    if camera is not None:
        camera.release()  # Release the camera
        camera = None


CLASS_COLOR = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (128, 128, 0),  # Olive
    (0, 128, 128),  # Teal
    (128, 128, 128)  # Gray
]

# Map class names to colors dynamically
class_color_map = {}


def detect_objects(frame):
    """Perform object detection, return annotated frame, and brake status."""
    global speed
    results = model(frame)
    annotated_frame = frame.copy()
    brake_message = None
    should_brake = False
    detected_objects = set()  # Store the types of detected objects

    # Define thresholds for reducing speed based on object proximity
    brake_threshold = 300  # If object box height is greater than this, apply brake
    slow_down_threshold = 150  # If object box height is greater than this, slow down

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for i, box in enumerate(boxes):
            confidence = confidences[i]
            class_id = class_ids[i]

            # Only process if confidence is higher than 0.5
            if confidence > 0.5:
                x1, y1, x2, y2 = map(int, box)
                # Get the label of the detected class
                label = model.names[class_id]
                detected_objects.add(label)

                # Assign a color to the class if it doesn't exist
                if label not in class_color_map:
                    class_color_map[label] = CLASS_COLOR[len(
                        class_color_map) % len(CLASS_COLOR)]

                # Get the color for the label
                color = class_color_map[label]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1),
                              (x2, y2), color, 2)

                # Display label and confidence
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # If a person, vehicle, or obstacle is detected, apply speed adjustments
                if label in ['person', 'cat', 'vehicle', 'obstacle', 'traffic signal', 'stop sign']:
                    box_height = y2 - y1  # Box height as a proxy for distance

                    if box_height > brake_threshold:
                        # Object is very close, press the brake
                        should_brake = True
                    elif box_height > slow_down_threshold:
                        # Object is relatively close, slow down
                        # Slow down to a minimum of 30 km/h
                        speed = max(speed - 20, 30)

    # Apply braking if necessary
    if should_brake:
        brake_message = "Pressing brake! Object is too close!"
        speed = 0  # Stop the car
    else:
        # If no dangerous objects, gradually increase speed
        brake_message = f"Speed: {speed} km/h"
        speed = min(speed + 5, 100)  # Increase speed up to 100 km/h

    return annotated_frame, brake_message


def generate_frames():
    start_camera()  # Ensure the camera is running
    while True:
        success, frame = camera.read()  # Read frame from the camera
        if not success:
            break
        else:
            # Detect objects and annotate frame
            detected_frame, message = detect_objects(frame)

            # Add message to frame (display speed/brake message)
            cv2.putText(detected_frame, message, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', detected_frame)
            frame = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def live_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


@require_http_methods(["POST"])
def stop_stream(request):
    stop_camera()  # Stop the camera when requested
    return JsonResponse({'status': 'Camera stopped'})


def speed_status(request):
    """Return the current speed or brake status."""
    global speed
    if speed == 0:
        message = "Pressing brake! Object is too close!"
    else:
        message = f"Speed: {speed} km/h"
    return JsonResponse({'message': message})


def simulation(request):
    return render(request, 'livefeed.html')

# Overview view
def overview(request):
    return render(request, 'overview.html')

# FAQ view
def faq(request):
    return render(request, 'faq.html')

# Tutorial view
def tutorial(request):
    return render(request, 'tutorial.html')
