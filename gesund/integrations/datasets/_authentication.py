import time

import keyring
import jwt
import requests
from keyrings.alt.file import PlaintextKeyring


class AuthService:
    @staticmethod
    def _save_token_to_keyring(token: str, keyring_service_name: str):
        """Store the token securely in keyring."""
        keyring.set_keyring(PlaintextKeyring())
        keyring.set_password(keyring_service_name, "auth_token", token)

    @staticmethod
    def _load_token_from_keyring(keyring_service_name: str = "gesund"):
        """Retrieve the token securely from keyring."""
        return keyring.get_password(keyring_service_name, "auth_token")

    @staticmethod
    def get_auth_token(user_name: str, password: str, base_url: str, keyring_service_name: str):
        """
        Get the authentication token using the provided user credentials.
        Auth token will be stored securely in keyring for future use.
        """
        data = {"user_name": user_name, "password": password}
        url = f"{base_url}/auth/login"
        response = requests.post(url, json=data)

        if response.status_code == 200:
            token = response.json().get('token')
            AuthService._save_token_to_keyring(token, keyring_service_name)
            return token
        else:
            raise Exception("Authentication failed")

    @staticmethod
    def _check_token_expired(auth_token: str) -> bool:
        """Check if the token is expired."""
        if not auth_token:
            return True

        try:
            payload = jwt.decode(auth_token, options={"verify_signature": False})
            exp = payload.get('exp')

            if exp is None:
                print("No expiration claim found in the token.")
                return True  # Treat it as expired if there's no 'exp' claim

            current_time = time.time()
            return current_time > exp
        except jwt.ExpiredSignatureError:
            return True
        except jwt.JWTError:
            print("Error decoding token.")
            return True

    @staticmethod
    def refresh_token(user_name: str, password: str, base_url: str, keyring_service_name: str):
        """Refresh the token if it's expired."""
        print("Token expired, re-authenticating...")
        return AuthService.get_auth_token(user_name, password, base_url, keyring_service_name)

    @staticmethod
    def make_authenticated_request(endpoint: str, method="GET", data=None, user_name=None, password=None, base_url=None, keyring_service_name=None):
        """Make an authenticated API call with the current token."""
        # Load token from keyring if not already provided
        auth_token = AuthService._load_token_from_keyring(keyring_service_name)

        # Check if the token is expired
        if AuthService._check_token_expired(auth_token):
            # If expired, refresh the token
            auth_token = AuthService.refresh_token(user_name, password, base_url, keyring_service_name)

        # Prepare headers for the API call
        headers = {"x-access-token": auth_token}

        url = f"{base_url}/{endpoint}"

        # Make the request based on the HTTP method
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError("Unsupported HTTP method")

        # Handle token expiration during the request
        if response.status_code == 401:
            print("Unauthorized, token might have expired. Re-authenticating...")
            auth_token = AuthService.refresh_token(user_name, password, base_url, keyring_service_name)
            headers = {"x-access-token": auth_token}
            
            # Retry the request with the new token
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)

        return response.json()
