/* eslint-disable react-hooks/exhaustive-deps */
import React, {useEffect, useState} from 'react';

interface ProgressImageProps {
	phace: 'thinking' | 'typing';
}

const ProgressImage: React.FC<ProgressImageProps> = ({phace}) => {
	const [maskProgress, setMaskProgress] = useState(30);

	useEffect(() => {
		const limit = phace === 'thinking' ? 66 : 100;
		if (maskProgress < limit) {
			setTimeout(() => {
				setMaskProgress((oldVal) => (phace === 'typing' && oldVal < 50 ? 50 : oldVal) + 1.5);
			}, 1000);
		}
	}, [maskProgress]);

	useEffect(() => {
		if (phace === 'typing') setMaskProgress(50);
	}, [phace]);

	return (
		<div style={{position: 'relative'}} className='bg-white rounded-full me-[8px] h-[36px] w-[36px]'>
			<img src={'parlant-bubble-muted.svg'} alt='Progress' height={36} width={36} className='opacity-[0.3] rounded-full absolute' />
			<img
				src='parlant-logo-after.svg'
				height={36}
				width={36}
				alt=''
				className='rounded-full absolute z-10'
				style={{
					clipPath: `inset(${100 - maskProgress}% 0 0 0)`,
					transition: 'clip-path 500ms',
					// objectFit: 'cover',
					// maskImage: `linear-gradient(to top, rgba(0, 0, 0, 1) ${maskProgress}%, rgba(0, 0, 0, 0.1) ${maskProgress + 10}%)`,
					// WebkitMaskImage: `linear-gradient(to top, rgba(0, 0, 0, 1) ${maskProgress}%, rgba(0, 0, 0, 0.1) ${maskProgress + 10}%)`,
				}}
			/>
		</div>
	);
};

export default ProgressImage;
