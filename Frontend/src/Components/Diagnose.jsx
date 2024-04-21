// import React, { useState } from 'react';

// import Navbar2 from "./Navbar2";


// const Diagnose = () => {
//     return (
//         <div>

//             <div
//                 style={{ height: '80vh', backgroundImage: "url('/bgimg4.png')", backgroundSize: 'contain', backgroundPosition: 'center' }}
//                 className="w-full flex flex-col items-center justify-center bg-cover bg-center after:bg-sky-300 after:-z-10 after:rounded-full after:-top-4 after:-right-4 after:blur-xl border after:[box-shadow:-150px_50px_10px_100px_#7dd3fc]" >
//                 <Navbar2 />
//             </div>
//             <div>

//                 <div className="relative overflow-hidden max-w-8xl h-80 rounded shadow flex flex-col justify-between p-3 after:absolute after:w-24 after:h-24 after:bg-sky-300 after:-z-10 after:rounded-full after:-top-4 after:-right-4 after:blur-xl border after:[box-shadow:-150px_50px_10px_100px_#7dd3fc]" style={{ paddingLeft: "200px", paddingRight: "200px" }} id="login">
//                     <form method="post" action="" className="text-gray-700">
//                         <div className="flex justify-between gap-4 mb-4">
//                             <div className="flex-1">
//                                 <label htmlFor="text" className="text-xl font-bold after:content-['*']">Name  </label>

//                                 <input
//                                     required
//                                     type="text"
//                                     className="relative bg-transparent border-2 border-neutral-500 placeholder-violet-700 focus:ring-violet-500 placeholder-opacity-60 focus:border-violet-500 w-full p-2 mb-2 mt-1 ring-none focus:outline-none focus:shadow-outline focus:shadow-violet-500/50"
//                                     style={{ boxShadow: '0 0 8px rgba(138, 43, 226, 0.6)' }} // Initial soft glow effect
//                                 />

//                             </div>
//                             <div className="flex-1" style={{ paddingLeft: '50px' }}>

//                                 <label htmlFor="email" className="text-xl font-bold after:content-['*']">Email </label>

//                                 <input
//                                     required
//                                     type="email"
//                                     className="relative bg-transparent border-2 border-neutral-500 placeholder-violet-700 focus:ring-violet-500 placeholder-opacity-60 focus:border-violet-500 w-full p-2 mb-2 mt-1 ring-none focus:outline-none focus:shadow-outline focus:shadow-violet-500/50"
//                                     style={{ boxShadow: '0 0 8px rgba(138, 43, 226, 0.6)' }} // Initial soft glow effect
//                                 />

//                             </div>
//                         </div>
//                         <div className=' upload w-100px h-50px cursor-pointer ' style={{
//                             display: 'flex', width: '18%', padding: '1%', alignItems: 'center', justifyItems: 'center', border: '2px solid rgba(138, 43, 226, 0.6)', position: 'relative',
//                             left: '41% ', fontSize: '30px', boxShadow: '0px 0px 100px rgb(1, 235, 252), inset 0px 0px 10px rgb(1, 235, 252), 0px 0px 5px rgb(255, 255, 255)', animation: 'flicker 2s linear infinite'
//                         }}>

//                             <input className="photoinput absolute opacity-0 w-full h-full cursor-pointer " name="file" type="file" style={{ position: 'relative', width: '100%', padding: '1%' }} />
//                             <svg
//                                 xmlns="http://www.w3.org/2000/svg"
//                                 width="2em"
//                                 height="1em"
//                                 strokeLinejoin="round"
//                                 strokeLinecap="round"
//                                 viewBox="0 0 24 24"
//                                 strokeWidth={2}
//                                 fill="none"
//                                 stroke="currentColor"
//                                 className="icon"
//                                 style={{
//                                     position: 'relative',
//                                     bottom: '2px',
//                                     right: '38%'
//                                 }}
//                             >
//                                 <polyline points="16 16 12 12 8 16" />
//                                 <line y2={21} x2={12} y1={12} x1={12} />
//                                 <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
//                                 <polyline points="16 16 12 12 8 16" />
//                             </svg>

//                         </div>
//                         {/* </div> */}
//                         <div className=' upload w-100px h-50px' style={{
//                             display: 'flex', width: '22%', padding: '1%', alignItems: 'center', justifyItems: 'center', position: 'relative',
//                             left: '40% ', fontSize: '30px'
//                         }}>
//                             <label htmlFor="x-ray" className="text-xl font-bold after:content-['*']">Upload Your X-Ray </label>
//                         </div>
//                         <div className=' upload w-100px h-50px cursor-pointer ' style={{
//                             display: 'flex', width: '18%', padding: '1%', alignItems: 'center', justifyItems: 'center', position: 'relative',
//                             left: '41% ', fontSize: '30px'
//                         }}>
//                             <button className="border-none w-80 h-10 rounded-full flex justify-center items-center gap-3 bg-gray-900 cursor-pointer transition-all duration-450 ease-in-out hover:bg-gradient-to-b hover:from-purple-400 hover:to-purple-700 hover:shadow-custom hover:-translate-y-1">
//                                 <svg
//                                     height="24"
//                                     width="24"
//                                     viewBox="0 0 24 24"
//                                     fill="gray"
//                                     className="fill-current transition-all duration-800 ease hover:fill-white hover:scale-110"
//                                     xmlns="http://www.w3.org/2000/svg"
//                                 >
//                                     <path d="M10,21.236,6.755,14.745.264,11.5,6.755,8.255,10,1.764l3.245,6.491L19.736,11.5l-6.491,3.245ZM18,21l1.5,3L21,21l3-1.5L21,18l-1.5-3L18,18l-3,1.5ZM19.333,4.667,20.5,7l1.167-2.333L24,3.5,21.667,2.333,20.5,0,19.333,2.333,17,3.5Z"></path>
//                                 </svg>
//                                 <span className="font-semibold text-gray-400 text-base hover:text-white">Diagnose</span>
//                             </button>
//                         </div>

//                     </form>
//                 </div>

//             </div>
//         </div>

//     );
// };

// export default Diagnose;



// WORKING CODE --------------------------------------------------------------------------------------==========================

// import React, { useState } from 'react';
// import axios from 'axios';

// function ImageUploader() {

//     const Logout =()=>{
//         localStorage.clear()
//         window.location.reload()
//         history.back()
//     }

//     const [selectedImage, setSelectedImage] = useState(null);
//     const [message, setMessage] = useState('');
//     const [showResult, setShowResult] = useState(false);

//     const handleImageChange = (event) => {
//         const file = event.target.files[0];
//         setSelectedImage(file);
//     };

//     const handleSubmit = async (event) => {
//         event.preventDefault();

//         if (!selectedImage) {
//             console.log('No image selected');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('file', selectedImage); // Changed from 'img' to 'file' to match the backend

//         try {
//             const response = await axios.post('http://127.0.0.1:8000/upload/', formData, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data'
//                 },
//                 responseType: 'json'  // Ensures response type is expected as JSON
//             });

//             setMessage(response.data.message);
//             setShowResult(true);
//         } catch (error) {
//             console.error('Error uploading image:', error);
//             if (error.response) {
//                 // You can access more error details here if needed
//                 console.error('Error response:', error.response);
//             }
//         }
//     };

//     return (
//         <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
//             <div className="bg-white rounded-lg shadow-lg p-8">
//                 <h1 className="text-3xl font-bold mb-6 text-center">Image Uploader</h1>
//                 <form onSubmit={handleSubmit} className="space-y-4">
//                     <label htmlFor="image-upload" className="block w-full mb-4">
//                         <div className="border-4 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-gray-400 transition duration-300 ease-in-out">
//                             <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto mb-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
//                                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
//                             </svg>
//                             <span className="text-gray-500">Select Image</span>
//                             <input
//                                 id="image-upload"
//                                 type="file"
//                                 accept="image/*"
//                                 className="hidden"
//                                 name='file'
//                                 onChange={handleImageChange}
//                             />
//                         </div>
//                     </label>
//                     {selectedImage && (
//                         <div className="flex justify-center">
//                             <img
//                                 src={URL.createObjectURL(selectedImage)}
//                                 alt="Selected Image"
//                                 className="max-h-64 rounded-lg"
//                             />
//                         </div>
//                     )}
//                     <button
//                         type="submit"
//                         className="w-full py-3 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition duration-300 ease-in-out"
//                     >
//                         Upload Image
//                     </button>
//                 </form>

//                 {showResult && (
//                     <div className="mt-6 text-center">
//                         <p className="font-semibold text-lg">Prediction:</p>
//                         <p className="mt-2 text-gray-700">{message}</p>
//                     </div>
//                 )}

//                 <div className="flex justify-end mt-6">
//                     <button onClick={Logout} className="bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-300 ease-in-out">
//                         Logout
//                     </button>
//                 </div>
//             </div>
//         </div>
//     );
// }

// export default ImageUploader;

// ---------------------------------------------------------------------------------------------------------------------------------------
// =================================================================================================================================
// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
// FULL FLEDGED RUNNING CODE WITH ONLY UPLOAD BUTTON

import axios from 'axios';
import { Result } from 'postcss';
import { useState } from 'react';

function ImageUploader() {

    const [showSeverity, setShowSeverity] = useState(false);

    const Logout = () => {
        localStorage.clear();
        window.location.reload();
        history.back();
    };

    const [selectedImage, setSelectedImage] = useState(null);
    const [message, setMessage] = useState('');
    const [confidence, setConfidence] = useState('');
    const [prediction, setPrediction] = useState('');
    const [showResult, setShowResult] = useState(false);

    const toggleSeverity = async () => {
        const formData = new FormData();
        formData.append('email', localStorage.getItem('email'));

        try {
            
            const response = await fetch('http://127.0.0.1:8000/explanation/', {
                method: 'POST',
                body: formData,
            })
            // setMessage(JSON.stringify(response.data));
            const data = await response.json();
            if (response.ok)
            {
                setShowSeverity(!showSeverity)
            }
            // console.log(response.data)
        } catch (error) {
            console.error(error);
            if (error.response) {
                console.error('Error response:', error.response);
            }
        }
    };


    const handleImageChange = (event) => {
        const file = event.target.files[0];
        setSelectedImage(file);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        if (!selectedImage) {
            console.log('No image selected');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedImage);
        formData.append('email', localStorage.getItem('email'));

        try {
            const response = await fetch('http://127.0.0.1:8000/upload/', {
                method: 'POST',
                body: formData,
            })
            // setMessage(JSON.stringify(response.data));
            const data = await response.json();
            console.log("Before Set Message")
            console.log(data)
            setMessage(data.class);
            setConfidence(data.confidence);
            setShowResult(true);
            // console.log(response.data)
        } catch (error) {
            console.error(error);
            if (error.response) {
                console.error('Error response:', error.response);
            }
        }
    };

    const handleImagePrediction = async () => {
        if (!selectedImage) {
            console.log('No image selected');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedImage);

        try {
            const response = await axios.post('http://127.0.0.1:8000/predict/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            setPrediction(`Class: ${response.data.class}, Confidence: ${response.data.confidence.toFixed(2)}`);
        } catch (error) {
            console.error('Error predicting image:', error);
            if (error.response) {
                console.error('Error response:', error.response);
            }
        }
    };


    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
            <div className="bg-white rounded-lg shadow-lg p-8">
                <h1 className="text-3xl font-bold mb-6 text-center">Image Uploader and Predictor</h1>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <label htmlFor="image-upload" className="block w-full mb-4">
                        <div className="border-4 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-gray-400 transition duration-300 ease-in-out">
                            <input
                                id="image-upload"
                                type="file"
                                accept="image/*"
                                className="hidden"
                                onChange={handleImageChange}
                            />
                            Select or Drop Image Here
                        </div>
                    </label>
                    {selectedImage && (
                        <div className="flex justify-center">
                            <img
                                src={URL.createObjectURL(selectedImage)}
                                alt="Selected Image"
                                className="max-h-64 rounded-lg"
                            />
                        </div>
                    )}
                    <div className="grid grid-cols-2 gap-4">
                        <button
                            type="submit"
                            // action="submit"
                            onClick={showResult}
                            className="py-3 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition duration-300 ease-in-out"
                        >
                            Upload Image
                        </button>
                        <button
                            type="button"
                            onClick={toggleSeverity}
                            className="py-3 px-4 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-300 ease-in-out"
                        >
                            Show Severity
                        </button>
                    </div>
                </form>


                {showResult && (
                    <>
                        <div className="mt-6 text-center">
                            <p className="font-semibold text-lg">Class:</p>
                            <p className="mt-2 text-gray-700">{message}</p>
                        </div>
                        <div className="mt-6 text-center">
                            <p className="font-semibold text-lg">Confidence :</p>
                            <p className="mt-2 text-gray-700">{confidence}</p>
                        </div>
                    </>
                )}

                {prediction && (
                    <div className="mt-6 text-center">
                        <p className="font-semibold text-lg">Prediction Result :</p>
                        <p className="mt-2 text-gray-700">{prediction}</p>
                    </div>
                )}

                {showSeverity && (
                    <div className="image-result">
                        <img src="final_image.png" alt="" />
                    </div>
                )}

                <div className="flex justify-end mt-6">
                    <button onClick={Logout} className="bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-300 ease-in-out">
                        Logout
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ImageUploader;



// +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

// import axios from 'axios';
// import { useState } from 'react';

// function ImageUploader() {
//     const [selectedImage, setSelectedImage] = useState(null);
//     const [message, setMessage] = useState('');
//     const [confidence, setConfidence] = useState('');
//     const [prediction, setPrediction] = useState('');
//     const [showResult, setShowResult] = useState(false);
//     const [showSeverityImage, setShowSeverityImage] = useState(false); // New state for toggling severity image

//     const Logout = () => {
//         localStorage.clear();
//         window.location.reload();
//         history.back();
//     };

//     const handleImageChange = (event) => {
//         const file = event.target.files[0];
//         setSelectedImage(file);
//     };

//     const handleSubmit = async (event) => {
//         event.preventDefault();

//         if (!selectedImage) {
//             console.log('No image selected');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('file', selectedImage);

//         try {
//             const response = await axios.post('http://127.0.0.1:8000/upload/', formData, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data'
//                 }
//             });
//             setMessage(response.data.class);
//             setConfidence(response.data.confidence);
//             console.log(response.data)
//             setShowResult(true);
//         } catch (error) {
//             console.error('Error uploading image:', error);
//             if (error.response) {
//                 console.error('Error response:', error.response);
//             }
//         }
//     };

//     const handleImagePrediction = async () => {
//         if (!selectedImage) {
//             console.log('No image selected');
//             return;
//         }

//         const formData = new FormData();
//         formData.append('file', selectedImage);

//         try {
//             const response = await axios.post('http://127.0.0.1:8000/predict/', formData, {
//                 headers: {
//                     'Content-Type': 'multipart/form-data'
//                 }
//             });

//             setPrediction(`Class: ${response.data.class}, Confidence: ${response.data.confidence.toFixed(2)}`);
//         } catch (error) {
//             console.error('Error predicting image:', error);
//             if (error.response) {
//                 console.error('Error response:', error.response);
//             }
//         }
//     };

//     const toggleSeverityImage = () => {
//         setShowSeverityImage(!showSeverityImage);
//     };

//     return (
//         <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
//             <div className="bg-white rounded-lg shadow-lg p-8">
//                 <h1 className="text-3xl font-bold mb-6 text-center">Image Uploader and Predictor</h1>
//                 <form onSubmit={handleSubmit} className="space-y-4">
//                     <label htmlFor="image-upload" className="block w-full mb-4">
//                         <div className="border-4 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-gray-400 transition duration-300 ease-in-out">
//                             <input
//                                 id="image-upload"
//                                 type="file"
//                                 accept="image/*"
//                                 className="hidden"
//                                 onChange={handleImageChange}
//                             />
//                             Select or Drop Image Here
//                         </div>
//                     </label>
//                     {selectedImage && (
//                         <div className="flex justify-center">
//                             <img
//                                 src={URL.createObjectURL(selectedImage)}
//                                 alt="Selected Image"
//                                 className="max-h-64 rounded-lg"
//                             />
//                         </div>
//                     )}
//                     <div className="grid grid-cols-2 gap-4">
//                         <button
//                             type="submit"
//                             className="py-3 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition duration-300 ease-in-out"
//                         >
//                             Upload Image
//                         </button>
//                         <button
//                             type="button"
//                             onClick={toggleSeverityImage}
//                             className="py-3 px-4 bg-green-500 text-white rounded-lg hover:bg-green-600 transition duration-300 ease-in-out"
//                         >
//                             Show Severity
//                         </button>
//                     </div>
//                 </form>

//                 {showSeverityImage && (
//                     <div className="flex justify-center mt-6">
//                         <img src ="final_img.png" alt="Severity" className="max-h-64 rounded-lg" />
//                     </div>
//                 )}

//                 {showResult && (
//                     <>
//                         <div className="mt-6 text-center">
//                             <p className="font-semibold text-lg">Class:</p>
//                             <p className="mt-2 text-gray-700">{message}</p>
//                         </div>
//                         <div className="mt-6 text-center">
//                             <p className="font-semibold text-lg">Confidence :</p>
//                             <p className="mt-2 text-gray-700">{confidence}</p>
//                         </div>
//                     </>
//                 )}

//                 {prediction && (
//                     <div className="mt-6 text-center">
//                         <p className="font-semibold text-lg">Prediction Result :</p>
//                         <p className="mt-2 text-gray-700">{prediction}</p>
//                     </div>
//                 )}

//                 <div className="flex justify-end mt-6">
//                     <button onClick={Logout} className="bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-300 ease-in-out">
//                         Logout
//                     </button>
//                 </div>
//             </div>
//         </div>
//     );
// }

// export default ImageUploader;

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



// ==================================================================================================================================
// PREVIOUSLY USED CODE
// import React, { useState } from 'react';

// const UploadForm = () => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [uploadStatus, setUploadStatus] = useState(null);

//   const handleFileChange = (event) => {
//     setSelectedFile(event.target.files[0]);
//   };

//   const handleSubmit = async (event) => {
//     event.preventDefault();
//     const formData = new FormData();
//     formData.append('file', selectedFile);

//     try {
//       const response = await fetch('/upload/', {
//         method: 'POST',
//         body: formData,
//       });

//       if (response.ok) {
//         const data = await response.json();
//         setUploadStatus(data.message);
//       } else {
//         throw new Error('Upload failed');
//       }
//     } catch (error) {
//       setUploadStatus('Upload failed: ' + error.message);
//     }
//   };

//   return (
//     <div>
//       <h1>Upload Image</h1>
//       <form onSubmit={handleSubmit}>
//         <input type="file" onChange={handleFileChange} required />
//         <button type="submit">Upload Image</button>
//       </form>
//       {uploadStatus && <p>{uploadStatus}</p>}
//     </div>
//   );
// };

// export default UploadForm;
