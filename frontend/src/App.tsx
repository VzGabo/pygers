import Navbar from './components/Navbar'
import Button from './components/Button'
import IconImage from './components/upload'
import { useState } from 'react'
// import  from './components/orden'

export default function App() {
  // crear usestate
  const [files, setFiles] = useState<File[]>([])
  // psar el ste file al boton como parametro
  // y qeu el bonotn le pase el set file a guardar
  return (
    <main>
      <Navbar />
      <main className='flex justify-center mt-10'>
        <IconImage />
        <Button background='primary' setFiles={setFiles}>
        </Button>
        <IconImage />
      </main>
    </main>
  )
}
