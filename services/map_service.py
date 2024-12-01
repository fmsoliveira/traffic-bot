import folium

class MapService:
    def __init__(self, location=(40.7128, -74.0060), zoom_start=12):
        self.location = location
        self.zoom_start = zoom_start

    def create_map(self):
        m = folium.Map(location=self.location, zoom_start=self.zoom_start)
        folium.Marker(self.location, popup='Location').add_to(m)
        return m._repr_html_()
