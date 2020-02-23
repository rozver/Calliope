from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .validators import validate_email


# Simple signup form with username, email and password
class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True, validators=[validate_email])

    class Meta:
        model = User
        widgets = {
            'username': forms.TextInput(attrs={'placeholder': 'Username'}),
            'password': forms.Textarea(
                attrs={'placeholder': 'Password'}),
        }
        fields = ('username', 'email', 'password1', 'password2',)
