import Home from './pages/home'
import { BrowserRouter, Routes, Route} from 'react-router-dom';
import CompGrabacion from './pages/Grabacion';


export default function App() {
  
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Home/>}></Route>
        <Route path="/Grabacion" element={<CompGrabacion />} />
        {/* <Route path="/Live" element={<Grabacion />} /> */}
      </Routes>
    </BrowserRouter>
  )
}

