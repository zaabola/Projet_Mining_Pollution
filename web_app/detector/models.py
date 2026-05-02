"""
Models for detector app - User profiles and management
"""

from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """Extended user profile with role management and approval workflow."""
    
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('employee', 'Employee'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='employee')
    is_approved = models.BooleanField(default=False)
    id_card_image = models.ImageField(upload_to='id_cards/', blank=True, null=True)
    generated_email = models.EmailField(blank=True, default='')
    extracted_first_name = models.CharField(max_length=100, blank=True, default='')
    extracted_last_name = models.CharField(max_length=100, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()} ({'Approved' if self.is_approved else 'Pending'})"
    
    class Meta:
        ordering = ['-created_at']
