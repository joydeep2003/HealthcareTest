import './App.css';
import React from "react";
import Navbar from './Components/Navbar';
import Hero from './Components/Hero';
import About from './Components/About';
import Testimonials from './Components/Testimonial';
import Footer from './Components/Footer';
import Diagnose from './Components/Diagnose';
import Stats from './Components/Stats';

function App() {
  return (
    <div className="w-full h-screen  background">
      <Navbar />
      <Hero />
      <Stats />
      <Footer />
     
    </div>
  );
}

export default App;