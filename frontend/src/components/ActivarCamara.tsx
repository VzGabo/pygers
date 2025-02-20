// import { useRef, useEffect } from 'react';

// export default function Camara() {
//   const videoRef = useRef(null);

//   useEffect(() => {
//     if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
//       navigator.mediaDevices.getUserMedia({ video: true })
//         .then(stream => {
//           if (videoRef.current) {
//             videoRef.current.srcObject = stream;
//             videoRef.current.play();
//           }
//         })
//         .catch(error => {
//           console.error('Error al acceder a la cámara:', error);
//         });
//     }
//   }, []);
// }


import React from 'react';

function Live() {
  return (
    <div>
      <h1>Live</h1>
      <p>Bienvenido a la página de Live</p>
    </div>
  );
}

export default Live