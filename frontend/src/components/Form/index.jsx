"use client";
import { useRef, useState } from "react";
import Button from "../Button";
import Image from "next/image";
import axios from "axios";

const Form = () => {
  const [text, setText] = useState("");
  const [image, setImage] = useState("/sample.png");
  const fileInputRef = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      const formData = new FormData();
      formData.append("image", file);
      try {
        const response = await axios.post(
          "http://127.0.0.1:8000/api/ocr/upload/",
          formData,
          {
            headers: { "Content-Type": "multipart/form-data" },
          }
        );
        if (response.data) {
          console.log(response.data);
          setText(response.data.text);
        }
      } catch (error) {
        console.error("Upload failed", error);
        alert("Image upload failed");
      }
    }
  };

  return (
    <div className="w-full mt-5 bg-[#f2f6ff] border border-[#f4f4f4] rounded-2xl p-5 flex gap-4">
      <div className="w-[324px] aspect-[324/392] bg-[#bbc5ff] relative">
        <Image
          src={image || "/sample.png"}
          fill
          alt="image showcase"
          className="object-contain"
        />
      </div>
      <div className="flex-1 flex flex-col">
        <div className="w-full flex justify-end items-center">
          <input
            type="file"
            accept="image/*"
            className="hidden"
            ref={fileInputRef}
            onChange={handleUpload}
          />
          <Button
            label={"Upload Image"}
            onClick={() => fileInputRef.current.click()}
          />
        </div>
        <textarea
          name=""
          id=""
          value={text}
          disabled={true}
          className="w-full flex-1 bg-white resize-none focus:outline-none p-5 rounded-lg mt-5"
        ></textarea>
      </div>
    </div>
  );
};

export default Form;
