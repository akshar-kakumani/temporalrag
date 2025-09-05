from django.urls import path
from . import views
 
urlpatterns = [
    path('', views.home, name='home'),
    path('stream/', views.stream_answer, name='stream_answer'),
    path('health/', views.health_check, name='health_check'),
    path('gemini/', views.gemini_web_answer, name='gemini_web_answer'),
] 