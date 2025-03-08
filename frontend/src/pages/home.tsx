import { useState } from 'react'
import Navbar from '../components/Navbar'
import Button from '../components/Button'
import Images from '../components/Images'
// import Live from './components/ActivarCamara';

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  
  return (
    <>
      <Navbar />
      {
          files.length === 0
          ?
          <Button 
            element='file'
            background='primary' 
            setFiles={setFiles}
          >
            Sube tus sospechosos
          </Button>
          :
          <Images 
            files={files} 
            setFiles={setFiles} 
          />
        }
  </>
  )
}

