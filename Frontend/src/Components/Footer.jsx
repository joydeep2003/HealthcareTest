import React from 'react';
import { FaInstagram, FaFacebookF, FaTwitter, FaYoutube } from 'react-icons/fa';



const Footer = () => {
  return (
    <div className="p-4 flex justify-around  items-center background border-t-2 border-stone-500" >
      {/* Left side */}
      <div className="flex space-x-4">
        <img src="public\logo.png" alt="Fortis" className="h-12" />
        <div>
          <p className="font-bold text-white-800 text-white">HealthLens</p>
          <address className="not-italic text-sm text-white " style={{ font: " rgb(255 255 255 )" }}>
            SRM University AP , Amravati , Andhra Pradesh
          </address>
        </div>
      </div>

      {/* Right side */}
      <div className="text-lg text-gray-600 text-white">STAY IN TOUCH
        <div className="flex  text-white px-2">
          <a className='px-1' href="#" aria-label="Instagram"><FaInstagram /></a>
          <a className='px-1' href="#" aria-label="Facebook"><FaFacebookF /></a>
          <a className='px-1' href="#" aria-label="Twitter"><FaTwitter /></a>
          <a className='px-1' href="#" aria-label="YouTube"><FaYoutube /></a>
        </div>
      </div>

    </div>
  );
};

export default Footer;



