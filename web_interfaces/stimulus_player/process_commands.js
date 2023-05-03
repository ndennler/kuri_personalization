var video_files = []
var audio_files = []


var video = document.getElementById('video-stimulus');
var source = document.getElementById('video-source');

var audio = new Audio('../data/aud/all_systems_down.wav');


/*

Code for changing video

*/
function switch_video_from_index(index){
    switch_video(video_files[index])
}
function switch_video(file){
    source.setAttribute('src', file);
    source.setAttribute('type', 'video/mp4');
    if (!video.paused) { // check if the video is playing
        video.pause(); // pause the video
        video.currentTime = 0; // set the currentTime to 0 to start from the beginning
      }
    video.load()
}
function play_video(){
    if (!video.paused) { // check if the video is playing
        video.pause(); // pause the video
        video.currentTime = 0; // set the currentTime to 0 to start from the beginning
      }
    video.play()
}


/*

Code for changing audio

*/
function switch_audio_from_index(index){
    switch_audio(audio_files[index])
}
function switch_audio(file){  
    audio = new Audio(file);
}
function play_audio(){
    if (!audio.paused) {
        audio.pause();
        audio.currentTime = 0;
      }
    audio.play()
}

function synch_stimulus(){
    video.load()
    video.play()
    if (!audio.paused) {
        audio.pause();
        audio.currentTime = 0;
      }
    audio.play()
}



//
//          Loading the CSV
//

// Create a new XMLHttpRequest object
var xhr = new XMLHttpRequest();

// Set the callback function to execute when the request is complete
xhr.onload = function() {
  // Parse the CSV data into an array of rows
  var rows = xhr.responseText.split('\n');

  // Loop through the rows and do something with each one
  for (var i = 0; i < rows.length; i++) {
    var row = rows[i].split(',');
    type = row[2]

    if(type == 'Video'){
        video_files.push('../data/vis/'+row[3])
    }
    if(type == 'Audio'){
        audio_files.push('../data/aud/'+row[3])
    }
  }
};

// Open the CSV file and send the request
xhr.open('GET', '../data/all_data.csv');
xhr.send();