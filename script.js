$.ajax({
    url: "http://localhost:8080/route",
    type: "GET",
    data: {
        point: ["25.396,68.361", "25.3965,68.362", "25.397,68.363", "25.398,68.364", "25.399,68.365", "25.4,68.366", "25.401,68.367"],
        profile: "car", // Add the profile
        format: "json" // Correct the format parameter
    },
    success: function (data) {
        console.log("Route fetched successfully:", data);
        animateTruck(data.paths[0].points.coordinates);
    },
    error: function (xhr) {
        console.error("Error fetching route:", xhr.responseText);
    }
});
