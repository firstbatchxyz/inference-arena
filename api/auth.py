import os
import httpx
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from .models import User, GitHubOAuthResponse, GitHubUserInfo, LoginResponse, UserTokens, TokenUpdateRequest, TokenResponse
from benchmark.mongo_client import Mongo
import json

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required. Generate with: openssl rand -hex 32")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Reduced from 30 to 15 minutes for better security

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")

# Validate required OAuth credentials
if not GITHUB_CLIENT_ID:
    raise ValueError("GITHUB_CLIENT_ID environment variable is required")
if not GITHUB_CLIENT_SECRET:
    raise ValueError("GITHUB_CLIENT_SECRET environment variable is required")

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

class AuthService:
    def __init__(self):
        self.mongo_client = Mongo(os.getenv("MONGODB_URL"))
    
    def save_user_tokens(self, github_id: int, token_type: str, token_value: str, expires_at: Optional[datetime] = None):
        """Save or update user tokens"""
        print(f"Debug: save_user_tokens called for github_id={github_id}, token_type={token_type}")
        
        # Check if user tokens exist
        existing_tokens = self.mongo_client.find_one("user_tokens", {"github_id": github_id})
        print(f"Debug: Existing tokens found: {existing_tokens is not None}")
        
        if existing_tokens:
            # Update existing tokens
            update_data = {
                f"{token_type}_token": token_value,
                "updated_at": datetime.utcnow()
            }
            
            if expires_at:
                update_data[f"{token_type}_token_expires_at"] = expires_at
            
            print(f"Debug: Updating existing tokens with data: {update_data}")
            self.mongo_client.update_one(
                "user_tokens",
                {"github_id": github_id},
                {"$set": update_data}
            )
        else:
            # Create new token record
            token_data = {
                "github_id": github_id,
                f"{token_type}_token": token_value,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            if expires_at:
                token_data[f"{token_type}_token_expires_at"] = expires_at
            
            print(f"Debug: Creating new token record with data: {token_data}")
            self.mongo_client.insert_one("user_tokens", token_data)
        
        print(f"Debug: Token save operation completed")
    
    def get_user_tokens(self, github_id: int) -> Optional[UserTokens]:
        """Get user tokens from database"""
        token_data = self.mongo_client.find_one("user_tokens", {"github_id": github_id})
        if token_data:
            return UserTokens(**token_data)
        return None
    
    def get_user_token(self, github_id: int, token_type: str) -> Optional[str]:
        """Get specific token for user"""
        token_data = self.mongo_client.find_one("user_tokens", {"github_id": github_id})
        if token_data:
            return token_data.get(f"{token_type}_token")
        return None
    
    def delete_user_token(self, github_id: int, token_type: str):
        """Delete specific token for user"""
        self.mongo_client.update_one(
            "user_tokens",
            {"github_id": github_id},
            {"$unset": {f"{token_type}_token": "", f"{token_type}_token_expires_at": ""}}
        )
    
    def is_token_valid(self, github_id: int, token_type: str) -> bool:
        """Check if token exists and is not expired"""
        token_data = self.mongo_client.find_one("user_tokens", {"github_id": github_id})
        if not token_data:
            return False
        
        token = token_data.get(f"{token_type}_token")
        if not token:
            return False
        
        # Check expiration if exists
        expires_at = token_data.get(f"{token_type}_token_expires_at")
        if expires_at and datetime.utcnow() > expires_at:
            return False
        
        return True
    
    def get_valid_token_for_service(self, github_id: int, service_name: str) -> Optional[str]:
        """Get a valid token for a specific service, with automatic refresh if needed"""
        token_value = self.get_user_token(github_id, service_name)
        
        if not token_value:
            return None
        
        # Check if token is still valid
        if not self.is_token_valid(github_id, service_name):
            # Token has expired, remove it
            self.delete_user_token(github_id, service_name)
            return None
        
        return token_value
    
    def refresh_github_token_if_needed(self, github_id: int) -> Optional[str]:
        """Refresh GitHub token if it's expired (GitHub tokens typically don't expire but this is a placeholder)"""
        # GitHub tokens typically don't expire unless revoked
        # This is a placeholder for future implementation if needed
        return self.get_valid_token_for_service(github_id, "github")

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def get_github_access_token(self, code: str) -> GitHubOAuthResponse:
        """Exchange authorization code for GitHub access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": GITHUB_REDIRECT_URI
                },
                headers={"Accept": "application/json"}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get access token from GitHub"
                )
            
            data = response.json()
            if "error" in data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"GitHub OAuth error: {data.get('error_description', data['error'])}"
                )
            
            return GitHubOAuthResponse(**data)
    
    async def get_github_user_info(self, access_token: str) -> GitHubUserInfo:
        """Get user information from GitHub using access token"""
        async with httpx.AsyncClient() as client:
            # Get basic user info
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info from GitHub"
                )
            
            user_data = response.json()
            
            # Get user's email addresses
            email_response = await client.get(
                "https://api.github.com/user/emails",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            if email_response.status_code == 200:
                emails = email_response.json()
                print(f"Debug: Found {len(emails)} email(s) for user {user_data.get('login', 'unknown')}")
                
                # Find the primary email or the first verified email
                primary_email = None
                for email_info in emails:
                    if email_info.get("primary", False):
                        primary_email = email_info.get("email")
                        print(f"Debug: Using primary email: {primary_email}")
                        break
                
                # If no primary email, use the first verified email
                if not primary_email:
                    for email_info in emails:
                        if email_info.get("verified", False):
                            primary_email = email_info.get("email")
                            print(f"Debug: Using verified email: {primary_email}")
                            break
                
                # If still no email, use the first email in the list
                if not primary_email and emails:
                    primary_email = emails[0].get("email")
                    print(f"Debug: Using first email: {primary_email}")
                
                # Update user data with the found email
                if primary_email:
                    user_data["email"] = primary_email
                else:
                    print(f"Debug: No email found for user {user_data.get('login', 'unknown')}")
            else:
                print(f"Debug: Failed to get emails from GitHub API: {email_response.status_code}")
                print(f"Debug: Email response: {email_response.text}")
            
            return GitHubUserInfo(**user_data)
    
    def get_or_create_user(self, github_user: GitHubUserInfo) -> User:
        """Get existing user or create new user from GitHub info"""
        # Check if user exists
        existing_user = self.mongo_client.find_one("users", {"github_id": github_user.id})
        
        if existing_user:
            # Update last login
            self.mongo_client.update_one(
                "users",
                {"github_id": github_user.id},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            return User(**existing_user)
        
        # Create new user
        if not github_user.email:
            # Log a warning that no email was found
            print(f"Warning: No email found for GitHub user {github_user.login} (ID: {github_user.id})")
            # You might want to handle this case differently based on your requirements
            # For now, we'll use a placeholder but you could make email required or handle it differently
        
        new_user = User(
            github_id=github_user.id,
            username=github_user.login,
            email=github_user.email or f"{github_user.login}@no-email-provided.com",
            avatar_url=github_user.avatar_url
        )
        
        # Save to database
        user_dict = new_user.dict()
        user_dict["created_at"] = new_user.created_at
        user_dict["last_login"] = new_user.last_login
        
        self.mongo_client.insert_one("users", user_dict)
        return new_user
    
    async def authenticate_github_user(self, code: str) -> LoginResponse:
        """Complete GitHub OAuth flow and return user with access token"""
        # Get access token from GitHub
        oauth_response = await self.get_github_access_token(code)
        
        # Get user info from GitHub
        github_user = await self.get_github_user_info(oauth_response.access_token)
        
        # Get or create user in our database
        user = self.get_or_create_user(github_user)
        
        # Save GitHub access token for later use
        # GitHub tokens typically don't expire unless revoked, but we'll store the scope info
        print(f"Debug: Saving GitHub token for user {user.github_id}")
        self.save_user_tokens(
            github_id=user.github_id,
            token_type="github",
            token_value=oauth_response.access_token
        )
        print(f"Debug: GitHub token saved successfully")
        
        # Create access token
        access_token = self.create_access_token(
            data={"sub": str(user.github_id), "username": user.username}
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user=user
        )

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token"""
    auth_service = AuthService()
    payload = auth_service.verify_token(credentials.credentials)
    github_id: str = payload.get("sub")
    
    if github_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    mongo_client = Mongo(os.getenv("MONGODB_URL"))
    user_data = mongo_client.find_one("users", {"github_id": int(github_id)})
    
    if user_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return User(**user_data)

# Optional dependency for endpoints that can work with or without authentication
async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security)) -> Optional[User]:
    """Get current user if authenticated, otherwise return None"""
    if credentials is None:
        return None
    
    try:
        auth_service = AuthService()
        payload = auth_service.verify_token(credentials.credentials)
        github_id = payload.get("sub")
        
        if github_id is None:
            return None
        
        # Get user from database
        mongo_client = Mongo(os.getenv("MONGODB_URL"))
        user_data = mongo_client.find_one("users", {"github_id": int(github_id)})
        
        if user_data is None:
            return None
        
        return User(**user_data)
    except (JWTError, HTTPException, Exception):
        return None
