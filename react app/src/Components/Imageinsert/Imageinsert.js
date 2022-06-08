import './Imageinsert.css'
import React from 'react';
import axios from 'axios';




function Navigation(){

const [image, setImage] = React.useState({ preview: "", raw: "" });


const handleChange = e => {
  if (e.target.files.length) {
    setImage({
      preview: URL.createObjectURL(e.target.files[0]),
      raw: e.target.files[0]
    });
  }
};

const [result, setResult] = React.useState("");



const handleSubmit = (e) => {
  e.preventDefault()
  // console.log(e)
  const formData = new FormData(e.target);
  console.log(formData)
  
  axios.post('http://127.0.0.1:5000/predict', formData)
  // axios.post('https://currency-detection.herokuapp.com/predict', formData)
  .then(function (response) {
    console.log(response);
    setResult(response.data)
  })
  .catch(function (error) {
    console.log(error);
  });
}



function clear(){
  setResult("");
  setImage({ preview: "", raw: "" });
}

  
  
  return (
    <div className="App">      
      <form onSubmit={handleSubmit} className="container mt-5 pt-5 pb-5" enctype="multipart/form-data">
          <div className="form-inline justify-content-center mt-5">
          
              <div className="input-group">
                  <input type="file" id="image" name="file" 
                  accept="image/*" onChange={handleChange} className="file-custom no-underline near-white bg-animate bg-navy hover-bg-near-black hover-gold inline-flex items-center ma2 tc br2 pa2 shadow-3 grow"/>
              </div>
              
              <label htmlFor="upload-button">
                {image.preview ? (
                  <img src={image.preview} alt="dummy" width="250" height="auto" />
                ) : (
                  <>
                    <span className="fa-stack fa-2x mt-3 mb-2">
                      <i className="fas fa-circle fa-stack-2x" />
                      <i className="fas fa-store fa-stack-1x fa-inverse" />
                    </span>
                    {/* <h5 className="text-center"></h5> */}
                  </>
                )}
              </label>
          </div>
          
          <ul>
            <li><span><button type="submit" className="butto">Predict</button></span></li>
          </ul>  
      </form>
      <ul className="image-item__btn-wrapper">
          <li><span><button  className="butto" onClick={() =>  clear()}>Remove</button></span></li>
      </ul>

      <article>
          <br/>
          <h2 >{result}</h2>
                    
       </article>
    </div>
  );
}

export default Navigation;