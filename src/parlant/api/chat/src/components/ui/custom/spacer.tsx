import {memo, ReactElement} from 'react';

const Spacer = (): ReactElement => {
	return <div className='w-[16px] min-w-[16px]'></div>;
};

export default memo(Spacer);
