import Tilt from 'react-tilt';
import './Logo.css';
import ima from './logo.png'

function Logo(){
return(
		<div className='ma3 mt0'>
			
			<Tilt className="Tilt " options={{ max : 25 }} style={{ height: 100, width: 150 }} >
				<div className="Tilt-inner pa3"> 
				 	<img style={{paddingTop: '5px'}}alt='logo' src={ima}/>
				</div>
			</Tilt>
			
		</div>


	);

}

export default Logo;