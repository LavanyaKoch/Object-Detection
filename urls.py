from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('contact/', views.contact, name='contact'),
    path('about/', views.about, name='about'),
    path('detect/', views.detect, name='detect'),
    path('live_feed/', views.live_feed, name='live_feed'),
    path('stop_stream/', views.stop_stream, name='stop_stream'),
    path('simulation/', views.simulation, name='simulation'),
    path('speed_status/', views.speed_status, name='speed_status'),
    path('overview/', views.overview, name='overview'),
    path('faq/', views.faq, name='faq'),
    path('tutorial/', views.tutorial, name='tutorial'),
]