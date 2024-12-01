function getLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      function (position) {
        // Get latitude and longitude from the geolocation API
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;

        // Set the values of the latitude and longitude fields
        document.getElementById("latitude").value = latitude;
        document.getElementById("longitude").value = longitude;
      },
      function (error) {
        alert("Unable to retrieve location. Please allow location access.");
      }
    );
  } else {
    alert("Geolocation is not supported by this browser.");
  }
}
