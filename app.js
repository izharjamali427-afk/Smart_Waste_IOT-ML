// Your Firebase configuration object (replace with your own credentials)
const firebaseConfig = {
 apiKey: "AIzaSyA3cCcxUA10Vvbdn5IpBrumruc9-CW_NGk",
      authDomain: "smartwaste-9b8d9.firebaseapp.com",
      databaseURL: "https://smartwaste-9b8d9-default-rtdb.asia-southeast1.firebasedatabase.app",
      projectId: "smartwaste-9b8d9",
      storageBucket: "smartwaste-9b8d9.firebasestorage.app",
      messagingSenderId: "416800460395",
      appId: "1:416800460395:web:8b3e618289982203d16b2e"
};

// Initialize Firebase
const app = firebase.initializeApp(firebaseConfig);
const database = firebase.database(app);

// Function to listen for real-time changes and fetch the most recent data for a bin
const listenForBinChanges = (binId, elementId, progressBarId) => {
  const binRef = database.ref(`bins/${binId}`);

  // Order by key and limit to the most recent entry
  binRef.orderByKey().limitToLast(1).on('child_added', (snapshot) => {
    if (snapshot.exists()) {
      const latestData = snapshot.val();
      const level = latestData;  // Get the most recent value (percentage)

      // Update the UI with the most recent level
      updateUI(elementId, progressBarId, level);
    } else {
      console.log(`${binId} data not found`);
    }
  }, (error) => {
    console.error(`Error fetching ${binId} data:`, error);
  });
};

// Function to update the UI elements with the fetched data
const updateUI = (elementId, progressBarId, level) => {
  const levelElement = document.getElementById(elementId);
  const progressBar = document.getElementById(progressBarId);

  levelElement.innerText = `${elementId.replace('-', ' ').toUpperCase()} Level: ${level}%`;
  progressBar.style.width = `${level}%`;

  // Change color based on level
  if (level >= 80) {
    progressBar.style.backgroundColor = 'red';
  } else if (level >= 50) {
    progressBar.style.backgroundColor = 'yellow';
  } else {
    progressBar.style.backgroundColor = 'green';
  }
};

// Initialize real-time data listeners for both bins
listenForBinChanges('bin1', 'bin1-level', 'bin1-progress');
listenForBinChanges('bin2', 'bin2-level', 'bin2-progress');
