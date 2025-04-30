document.getElementById("upload-form").onsubmit = async function (e) {
    e.preventDefault();

    let formData = new FormData();
    formData.append("source", document.getElementById("source").files[0]);
    formData.append("background", document.getElementById("background").files[0]);

    let response = await fetch("/upload/", {
        method: "POST",
        body: formData
    });

    let result = await response.json();
    console.log(result);
    if (result.output_path) {
        document.getElementById("output-image").src = result.output_path;
    }
};
