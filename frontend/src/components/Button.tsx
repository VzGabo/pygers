import React from 'react'
import UploadIcon from '../assets/icons/Upload'

interface Props {
  children: React.ReactNode,
  background: "primary" | "secondary"
}

export default function Button({ children, background }: Props) {
  return (
    <button className={`bg-${background} p-4 text-white rounded-2xl flex gap-2 cursor-pointer transition duration-200 hover:bg-primary-light active:scale-90`}>
      <UploadIcon color='white' /> 
      <p>{children}</p>
    </button>
  )
}
