from pydantic import BaseModel, Field
from typing import Any, Dict

class WeatherOutput(BaseModel):
    city: str = Field(..., description="The name of the city")
    temperature: float = Field(..., description="The current temperature in Celsius")
    humidity: float = Field(..., description="The current humidity percentage")
    wind_speed: float = Field(..., description="The wind speed in km/h")

__all__ = ["WeatherOutput"]