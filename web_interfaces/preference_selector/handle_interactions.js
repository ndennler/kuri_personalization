var stimulus_types = ['Video', 'Audio', 'Movement']
var vis_id, aud_id, kin_id;
var clickCount = 0;
var clickTimeout;

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
        } else if((i - 1) % cols === cols-1){ // check if this is the last button in the row
            button.style.width = "10vw";
            button.style.margin = "auto"
            button.style.height = "90%"
            button.textContent = `I don't like \nany of these\n${stimulus_types[row_num]}s`;
            button.id = `NA${stimulus_types[row_num]}`
        } else {
            button.style.width = 85/(cols-1) + "vw";
            button.textContent = `${stimulus_types[row_num]} ${i - row_num*cols - 1}`;
            button.id = `${stimulus_types[row_num]}${i - row_num*cols - 1}`
        }
        
        buttonGrid.appendChild(button);
    }
}

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

    //store selection TODO: actually do this
    if (type == 'Video'){
        vis_id = 0
    } else if (type == 'Audio') {
        aud_id = 0
    } else if (type == 'Movement'){
        kin_id = 0
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
}

//what to do if the button is single clicked
function playOption(event){
    console.log("Button single clicked");
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
        for (let i = 1; i <= 20; i++) {
        optionsList.appendChild(createOption(`Visual ${i}`));
        }
    } else if (group === 'sound') {
        document.getElementById('audiobtn').className += ' active'
        for (let i = 1; i <= 20; i++) {
        optionsList.appendChild(createOption(`Sound ${i}`));
        }
    } else if (group === 'movement') {
        document.getElementById('movementbtn').className += ' active'
        for (let i = 1; i <= 20; i++) {
        optionsList.appendChild(createOption(`Movement ${i}`));
        }
    }
}

function createOption(text) {
    var li = document.createElement('button');
    li.textContent = text;
    return li;
    }




// create the button grid with 3 rows and 4 columns
createButtonGrid(3, 5);
showOptions('visual')

