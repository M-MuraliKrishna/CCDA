
import './App.css';
import Navigation from "./Components/Navigation/Navigation";
import Logo from "./Components/Logo/Logo";
import Imageinsert from "./Components/Imageinsert/Imageinsert";
import Particles from 'react-particles-js';
import Predict from "./Components/Predict/Predict";



function App() {
  return (
    <div className="App">
      {/* <Navigation /> */}
      <Predict/>
      <Particles className='particles'
        params={{
          particles: {
            line_linked:{
              shadow: {
                enable: true,
                blur: 5
              }
            },
            number: {
              value: 50,
              density: {
                enable: true,
                value_area: 1000,
              }               
            },
          },
        }}
      />
      {/* <Logo /> */}
      <Imageinsert/>
      
    </div>
  );
}

export default App;
