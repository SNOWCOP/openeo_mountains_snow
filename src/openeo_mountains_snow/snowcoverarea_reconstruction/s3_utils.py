"""
S3 utilities for checkpoint and artifact management.

Handles S3 authentication, credential management, and file operations.
"""

import datetime
import hashlib
import boto3
import s3fs
from typing import Dict, Any


class S3Manager:
    """Manages S3 operations for checkpoint storage and retrieval."""
    
    def __init__(self, region: str = "waw3-1"):
        """
        Initialize S3 manager with AWS STS credentials.
        
        Args:
            region: AWS region code (default: waw3-1)
        """
        self.region = region
        self.otc_prod_url = f"https://openeo.prod.{region}.openeo-int.v1.dataspace.copernicus.eu/openeo/"
        self.sts_url = f"https://sts.{region}.openeo.v1.dataspace.copernicus.eu"
        self.s3_url = self.sts_url.replace('sts', 's3')
        self.bucket_name = f"openeo-artifacts-{region}"
        self.s3_endpoint = f"https://s3.{region}.openeo.v1.dataspace.copernicus.eu"
        
        self.credentials = None
        self.s3_client = None
        self.fs = None
        self.upload_prefix = None
    
    def authenticate(self) -> None:
        """Authenticate with openEO and get AWS STS credentials."""
        import openeo
        
        connection = openeo.connect(self.otc_prod_url).authenticate_oidc()
        
        token_parts = connection.auth.bearer.split('/')
        sts = boto3.client("sts", endpoint_url=self.sts_url)
        role_arn = f"arn:openeo:iam:::role/openeo-artifacts-{self.region}"
        
        response = sts.assume_role_with_web_identity(
            RoleArn=role_arn,
            RoleSessionName='snowcop-reconstruction',
            WebIdentityToken=token_parts[2],
            DurationSeconds=3600*12
        )
        
        assert 'Credentials' in response, f"Invalid credentials response: {response}"
        self.credentials = response['Credentials']
        
        # Initialize S3 client and filesystem
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.s3_url,
            aws_access_key_id=self.credentials['AccessKeyId'],
            aws_secret_access_key=self.credentials['SecretAccessKey'],
            aws_session_token=self.credentials['SessionToken']
        )
        
        self.fs = s3fs.S3FileSystem(
            key=self.credentials['AccessKeyId'],
            secret=self.credentials['SecretAccessKey'],
            token=self.credentials['SessionToken'],
            client_kwargs={'endpoint_url': self.s3_endpoint}
        )
        
        self._set_upload_prefix(response['SubjectFromWebIdentityToken'].encode())
    
    def _set_upload_prefix(self, subject_token: bytes) -> None:
        """Generate S3 upload prefix based on subject token and current date."""
        user_prefix = hashlib.sha1(subject_token).hexdigest()
        date_str = datetime.datetime.utcnow().strftime('%Y/%m/%d')
        self.upload_prefix = f"{user_prefix}/{date_str}/"
    
    def get_checkpoint_config(self, checkpoint_id: str = None) -> Dict[str, Any]:
        """
        Get checkpoint configuration for UDF context.
        
        Args:
            checkpoint_id: Optional checkpoint identifier for S3 path
            
        Returns:
            Dictionary with S3 credentials and configuration
        """
        if not self.credentials:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        if checkpoint_id is None:
            checkpoint_id = datetime.datetime.now().strftime("%H%M%S")
        
        return {
            "access_key": self.credentials['AccessKeyId'],
            "secret_key": self.credentials['SecretAccessKey'],
            "token": self.credentials['SessionToken'],
            "bucket": self.bucket_name,
            "prefix": f"{self.upload_prefix}{checkpoint_id}",
            "endpoint": self.s3_endpoint,
        }
    
    def print_credentials_info(self) -> None:
        """Print credential information for debugging."""
        if not self.credentials:
            print("Not authenticated yet.")
            return
        
        print(f"Access Key: {self.credentials['AccessKeyId']}")
        print(f"Bucket: {self.bucket_name}")
        print(f"Upload Prefix: {self.upload_prefix}")
