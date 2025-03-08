import Home from './pages/home'
import { BrowserRouter, Routes, Route} from 'react-router-dom';
import CompGrabacion from './pages/Grabacion';
import Decision from './components/Decision';
import WebcamComponent from './components/ActivarCamara'


export default function App() {
  
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Home/>}></Route>
        <Route path="/Grabacion" element={<CompGrabacion />} />
        <Route path="/Opciones" element={<Decision />} />
        <Route path="/Live" element={<WebcamComponent />} />
      </Routes>
    </BrowserRouter>
  )
}

