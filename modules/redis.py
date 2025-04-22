import streamlit as st
from streamlit.connections import BaseConnection
import redis
import json

class RedisConnection(BaseConnection[redis.Redis]):
    def _connect(self, **kwargs) -> redis.Redis:
        try:
            # Use connection parameters from secrets.toml by default
            params = {}
            if self._secrets:
                # Convert secrets to a regular dictionary instead of using copy()
                params.update(self._secrets)
            
            # Override with any kwargs provided
            params.update(kwargs)
            # Validate required parameters
            if "host" not in params:
                raise ValueError("Redis host is required")
            if "port" not in params:
                raise ValueError("Redis port is required")
                
            return redis.Redis(
                host=params.get("host"),
                port=int(params.get("port")),  # Ensure port is an integer
                password=params.get("password", ""),
                ssl=params.get("ssl", True),
                decode_responses=params.get("decode_responses", True)
            )
        except Exception as e:
            st.error(f"Redis connection error: {str(e)}")
            # Return a dummy connection or raise the error
            raise
    
    def get(self, key):
        return self._instance.get(key)
    
    def set(self, key, value, ttl=None):
        if ttl:
            return self._instance.setex(key, ttl, value)
        return self._instance.set(key, value)
    
    def get_json(self, key):
        try:
            data = self.get(key)
            return json.loads(data) if data else None
        except redis.exceptions.ConnectionError:
            # Try to reconnect
            self.reset()
            data = self.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            # Log the error
            st.error(f"Redis get_json error: {str(e)}")
            return False
    
    def set_json(self, key, data, ttl=None):
        try:
            json_data = json.dumps(data)
            if ttl:
                return self._instance.setex(key, ttl, json_data)
            return self._instance.set(key, json_data)
        except redis.exceptions.ConnectionError:
            # Try to reconnect once
            self.reset()
            json_data = json.dumps(data)
            if ttl:
                return self._instance.setex(key, ttl, json_data)
            return self._instance.set(key, json_data)
        except Exception as e:
            # Log the error
            st.error(f"Redis set_json error: {str(e)}")
            return False

@st.cache_resource
def get_redis_connection():
    return st.connection("redis", type=RedisConnection)
