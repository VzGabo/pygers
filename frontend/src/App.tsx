import { useState } from 'react'
import Navbar from './components/Navbar'
import Button from './components/Button'
import Images from './components/Images'

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  
  return (
    <main>
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
    </main>
  )
}

