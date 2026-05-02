"""
Tests for detector app
"""

from django.test import TestCase, Client
from django.urls import reverse
import os

class DetectorViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
    
    def test_index_page(self):
        """Test that index page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_models_endpoint(self):
        """Test that models endpoint returns JSON"""
        response = self.client.get('/api/models/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('available_models', data)
        self.assertIn('loaded_models', data)
