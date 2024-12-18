from django.urls import path
from . import views

urlpatterns = [
    path('upload/',views.process_image,name='process_image')
]