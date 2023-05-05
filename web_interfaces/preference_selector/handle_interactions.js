var stimulus_types = ['Video', 'Audio', 'Movement']
var video_ims = {}; var audio_ims = {}; var movement_ims = {}
var selectedTopSignalsPane = 'Video';
var vis_id, aud_id, kin_id;
var clickCount = 0;
var clickTimeout;


/*
This section is for UI elements when you first run it
*/

//switches between tabs
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";

    requestStimuliPub.publish({data: selectedTopSignalsPane + ': '})
}

document.getElementsByClassName("tablinks")[0].click();

// function to create the button grid dynamically
function createButtonGrid(rows, cols) {
    const buttonGrid = document.getElementById("button-grid");
    buttonGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    buttonGrid.style.gridTemplateRows = `repeat(${rows}, 1fr)`;

    for (let i = 1; i <= rows * cols; i++) {
        const button = document.createElement("button");
        row_num = Math.floor((i-1)/cols)

        button.ondblclick = chooseOption;
        button.onclick = handleClickCounts;

        if ((i - 1) % cols === 0) { // check if this is the first button in the row
            button.style.width = "9vw"; // set a different width for the first button
            button.style.margin = "auto"
            button.style.height = "90%"
            button.textContent = `No\n${stimulus_types[row_num]}`;
            button.id = `No${stimulus_types[row_num]}`
            button.dataset.index = -2
        } else if((i - 1) % cols === cols-1){ // check if this is the last button in the row
            button.style.width = "10vw";
            button.style.margin = "auto"
            button.style.height = "90%"
            button.textContent = `I don't like \nany of these\n${stimulus_types[row_num]}s`;
            button.id = `NA${stimulus_types[row_num]}`
            button.dataset.index = -1
        } else {
            button.style.width = 85/(cols-1) + "vw";
            button.textContent = `${stimulus_types[row_num]} ${i - row_num*cols - 1}`;
            button.id = `${stimulus_types[row_num]}${i - row_num*cols - 1}`
        }
        
        buttonGrid.appendChild(button);
    }
}



/*
This section is for doing the main choice interactions
*/



function handleClickCounts(id){
    clickCount++;
    clearTimeout(clickTimeout);
    clickTimeout = setTimeout(function() {
      if (clickCount === 1) {
        // Single click action
        playOption(id)
        selectButton(id)
      }
      clickCount = 0;
    }, 200);
}



//what to do if the button is double clicked (people find that option the most appealing)
function chooseOption(event){
    clearTimeout(clickTimeout);
    clickCount = 0;

    var type; const id = event.target.id
    stimulus_types.forEach(stimulus_type => {
        if(id.includes(stimulus_type)){
            type = stimulus_type
        }
    })
    
    selectButton(event)
    if(id.includes('No')) return

    //determine the index that the person chose
    let choice = 0
    if(id.includes('NA')){
        choice = 3
    } else {
        choice = parseInt(id.slice(-1) - 1)
    }

    //send the choice
    if (type == 'Video'){
        visualChoicePub.publish({data: choice})
    } else if (type == 'Audio') {
        auditoryChoicePub.publish({data: choice})
    } else if (type == 'Movement'){
        kinestheticChoicePub.publish({data: choice})
    }

    // Double click action
    console.log("Button double clicked", choice);

}

function selectButton(event){
    var type; const id = event.target.id

    stimulus_types.forEach(stimulus_type => {
        if(id.includes(stimulus_type)){
            type = stimulus_type
        }
    })
    console.log(type)
    if (type == 'Video'){
        vis_id = +event.target.dataset.index
        console.log(vis_id)
        visualStimPub.publish({data: vis_id})
    } else if (type == 'Audio') {
        aud_id = +event.target.dataset.index
        auditoryStimPub.publish({data: aud_id})
    } else if (type == 'Movement'){
        kin_id = +event.target.dataset.index
        kinestheticStimPub.publish({data: kin_id})
    }
    const buttonGroup = document.querySelectorAll(`[id*="${type}"]`);

    // Iterate over the buttons and remove the active class
    buttonGroup.forEach((button) => {
        if (button.classList.contains("active")) {
        button.classList.remove("active");
        }
        if(button.id == id){
            button.classList.add("active")
        } 
    });

    // playSignalPub.publish({data: type})
}

//what to do if the button is single clicked
function playOption(event){
    console.log("Button single clicked");
}



    
    
/*

This section is for showing the different options in the select tab

*/


function showOptions(group) {
    var optionsList = document.getElementById('options');
    
    // Clear the current options and the search bar
    optionsList.innerHTML = '';
    document.getElementById("search-bar").value = "";

    //set the buttons to inactive
    document.getElementById('visualbtn').className = document.getElementById('visualbtn').className.replace(" active", "");
    document.getElementById('audiobtn').className = document.getElementById('audiobtn').className.replace(" active", "");
    document.getElementById('movementbtn').className = document.getElementById('movementbtn').className.replace(" active", "");

    // Add the new options based on the selected group
    if (group === 'visual') {
        document.getElementById('visualbtn').className += " active"
        selectedTopSignalsPane = 'Video'
    } else if (group === 'sound') {
        document.getElementById('audiobtn').className += ' active'
        selectedTopSignalsPane = 'Audio'
    } else if (group === 'movement') {
        document.getElementById('movementbtn').className += ' active'
        selectedTopSignalsPane = 'Movement'
    }
    requestStimuliPub.publish({data: selectedTopSignalsPane + ': '})
}

//function to fill in the top however many best buttons
function populate_options(group, indices){
    var optionsList = document.getElementById('options');
    optionsList.innerHTML = '';
    // Add the new options based on the selected group
    if (group === 'Video') {
        for (let i = 0; i < indices.length; i++) {
            optionsList.appendChild(createOption(`${video_ims[indices[i]]}`, "", indices[i], 'Video'));
        }
    } else if (group === 'Audio') {
        for (let i = 0; i < indices.length; i++) {
            optionsList.appendChild(createOption(`${audio_ims[indices[i]].file}`, audio_ims[indices[i]].name, indices[i], 'Audio'));
        }
    } else if (group === 'Movement') {
        for (let i = 0; i < indices.length; i++) {
            optionsList.appendChild(createOption(`${movement_ims[indices[i]]}`, "", indices[i], 'Movement'));
        }
    }
}

function createOption(image_path, text, index, type) {
    var li = document.createElement('button');
    li.style.backgroundImage = `url("${image_path}")`
    li.style.paddingTop = "10%"
    li.textContent = text.replace(/_/g, ' ');
    li.dataset.index = index
    li.onclick = event => { selectButton(event); playOption(event)}
    li.id = type + ' top-choices ' + index //try to avoid accidental id collision with the other page
    return li;
    }

//attach keyup response to text box
document.getElementById("search-bar")
    .addEventListener("keyup", function(event) {
    if (event.key === 'Enter') {
        console.log(event.target.value)
        requestStimuliPub.publish({data: selectedTopSignalsPane + ':' + event.target.value})
    }
});

/*

This section is for finalizing the design

*/

function sendPlaySignal(){
    playSignalPub.publish({data: 'all'})
}

//shows confirmation (attached to the submit button)
function showConfirmation() {
    var confirmationDialog = document.getElementById('confirmation-background');
    var confirmationMessage = document.querySelector('.confirmation-message');

    confirmationMessage.textContent = 'Are you sure you want to want this signal?';
    confirmationDialog.style.visibility = 'visible';
    }

function confirmAction(actionConfirmed) {
    var confirmationDialog = document.getElementById('confirmation-background');

    if (actionConfirmed) {
        // User clicked "Yes"
        // Do something here, like submit a form or navigate to a new page
        signalDonePub.publish({data: ''+vis_id+','+aud_id+','+kin_id})
    } else {
        // User clicked "No"
        // Do nothing or provide feedback to the user
    }

    confirmationDialog.style.visibility= 'hidden';
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
        video_ims[row[1]] = '../data/vis/'+row[3].replace('.mp4', '.jpg')
    }
    if(type == 'Audio'){
        audio_ims[row[1]] = {file: '../data/aud/'+row[3].replace('.wav', '.jpg'), name: row[4]}
    }
    if(type == 'Movement'){
        movement_ims[row[1]] = '../data/kin/'+row[1]+'.png'
    }
  }
};

// Open the CSV file and send the request
xhr.open('GET', '../data/all_data.csv');
xhr.send();

// create the button grid with 3 rows and 4 columns
createButtonGrid(3, 5);
showOptions('visual')

