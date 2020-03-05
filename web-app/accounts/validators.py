from django.core.exceptions import ValidationError
from django.contrib.auth.models import User


# Check if email is already used
def validate_email(value):
    exists = User.objects.filter(email=value)

    # If it is already used, raise an error
    if exists:
        raise ValidationError("This email has already been used!")
