import googlemaps

def get_map_image(latitude, longitude):
    """Fetches a Google Map image for the given coordinates."""
    API_KEY = "AIzaSyAuaudnqm7MXjoIAYbfwgXyW3pbWH4FyoE"
    gmaps = googlemaps.Client(key=API_KEY)
    url = gmaps.static_map(
        center=(latitude, longitude),
        zoom=17, # Adjust the zoom level as needed
        size=(640, 480), # Adjust the image size as needed
        maptype="satellite", # Use "roadmap" for a standard map view
    )
    return url