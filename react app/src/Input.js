function Input(){
	function handleImageUpload() 
		{

		var image = document.getElementById("upload").files[0];

		    var reader = new FileReader();

		    reader.onload = function(e) {
		      document.getElementById("display-image").src = e.target.result;
		    }

		    reader.readAsDataURL(image);

		} 


return(
		<div >
			<input id="upload" type="file" onChange="handleImageUpload()" />
			<img id="display-image" src="" width="150" height="150" />
		
		</div>


	);

}

export default Input;