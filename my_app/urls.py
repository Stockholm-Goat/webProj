from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_page, name="main"),
    path('login/', views.log_in, name="login"),
    path('home/', views.home, name="home"),
    path('register/', views.register, name='register'),
    path('new_experiments', views.new_experiment, name='new_experiments'),
    path('experiment_page/<int:experiment_id>/', views.experiment_page, name='experiment_page'),
    path('process_block/<int:block_id>/<int:ex_id>/', views.process_block, name='process_block'),
]
