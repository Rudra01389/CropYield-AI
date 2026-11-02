"""Demo farmer account(s) for the CropYield AI login screen.

*** DEMONSTRATION ONLY - NOT PRODUCTION-GRADE SECURITY ***
This module exists so credentials live in one place instead of being scattered through app.py. The password is kept
as a salted PBKDF2-HMAC-SHA256 hash (not plain text) and checked in src/auth.py, but this is still a single shared
demo account with no database, no password reset, no rate limiting and no external identity provider. Do not reuse
this approach to protect real data.

To change the demo password, generate a new salt/hash:
    python -c "import hashlib,secrets;s=secrets.token_hex(16);print(s,hashlib.pbkdf2_hmac('sha256',b'NEWPASSWORD',bytes.fromhex(s),200000).hex())"
"""

PBKDF2_ITERATIONS = 200000

DEMO_USERS = {
    "samplefarmer": {
        "salt": "a686825ea612ff8a0d95367b45e4fbbd",
        "password_hash": "7a4f21b4f8f01fbe6007a7c1d9b3c0eb8d3cb63a8e921cb25e31cc695390c5ca",
        "profile": {
            "name": "Ramesh Patil",
            "district": "Pune",
            "state": "Maharashtra",
            "crop": "Rice",
            "season": "Kharif",
            "land_area_ha": 2.5,
        },
    },
}

# Used by the one-click "Continue as Demo Farmer" button (src/auth.py: demo_login). Deliberately not secret:
# this is a public demonstration account and the login screen has no username/password fields.
DEMO_CREDENTIALS_HINT = {"username": "samplefarmer", "password": "Farmer@123"}
