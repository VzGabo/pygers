import {guardar} from '../utils/onSubmitImage'

interface Props {
  background: "primary" | "secondary",
  setFiles: React.Dispatch<React.SetStateAction<File[]>>,
}

export default function Button({ background,setFiles }: Props) {
  return (
    <>
      <input 
          type='file' 
          className={`bg-${background} p-3 text-white rounded-2xl flex gap-2 cursor-pointer transition duration-200 hover:bg-primary-light active:scale-90`} 
          accept="image/*"
          onChange={(event) => guardar({setFiles, event})}
        />
    </>
  )
}
