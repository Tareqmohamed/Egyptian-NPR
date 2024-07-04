// server.js
const express = require('express');
const bodyParser = require('body-parser');

const app = express();
const port = 3000;

app.use(bodyParser.json());

app.post('/api/videos/plates', (req, res) => {
    
    const { plateNumber, captureDate, cameraId, platePath, carPath } = req.body;
    console.log(`Plate Number: ${plateNumber}`);
    console.log(`Capture Date: ${captureDate}`);
    console.log(`Camera ID: ${cameraId}`);
    console.log(`File Path: ${platePath}`);
    console.log(`File Path: ${carPath}`);
    console.log('-----------------------------------------------------------')
    res.send('Data received: ');
});

app.post('/api/images/plates', (req, res) => {
    const { plateNumber, captureDate } = req.body;

    console.log(`Plate Number: ${plateNumber}`);
    console.log(`Capture Date: ${captureDate}`);
    console.log('-----------------------------------------------------------')

    // Respond to the request indicating data has been received
    res.send('Data received: ');
});


app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});

