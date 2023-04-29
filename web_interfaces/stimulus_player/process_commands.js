

var video = document.getElementById('video-stimulus');
var source = document.getElementById('video-source');

var audio = new Audio('../data/aud/all_systems_down.mp3');


function switch_video(file){
    

    source.setAttribute('src', file);
    source.setAttribute('type', 'video/mp4');

}



function switch_audio(file){  
    audio = new Audio(file);
}

function synch_stimulus(){
    video.load()
    video.play()
    audio.play()
}