"""
hazard logic for edgevision

this file will contain helper functions for checking whether detected
objects are hazards, classifying urgency levels, and ranking alerts
before they are sent to the audio receiver.

"""

# classifies objects based on how critical they are to the user
def classify_urgency(distance_m: float, limit: float) -> str:
    ratio = distance_m / limit

    if ratio <= 0.25:
        return "critical"
    if ratio <= 0.60:
        return "high"
    return "low"


def direction_of(cx: float, width: int) -> str:
    # translate the location of the bounding box into direction
    if cx < width * 0.4:
        return "on your left"
    if cx > width * 0.6:
        return "on your right"
    return "ahead"