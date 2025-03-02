import {memo, ReactElement} from 'react';

const Spacer = (): ReactElement => {
	return <div className='w-[24px]'></div>;
};

export default memo(Spacer);
