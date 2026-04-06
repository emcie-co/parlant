import {useState, ReactNode} from 'react';
import {Dialog, DialogContent, DialogHeader, DialogPortal} from '@/components/ui/dialog';
import {DialogDescription, DialogTitle} from '@radix-ui/react-dialog';
import {spaceClick} from '@/utils/methods';
import clsx from 'clsx';

interface UseDialogReturn {
	openDialog: (title: string | null, content: ReactNode, dimensions: Dimensions) => void;
	DialogComponent: () => JSX.Element;
	closeDialog: (e?: React.MouseEvent) => void;
}

export interface Dimensions {
	height: string;
	width: string;
}

export const useDialog = (): UseDialogReturn => {
	const [dialogTitle, setDialogTitle] = useState<ReactNode>(null);
	const [dialogContent, setDialogContent] = useState<ReactNode>(null);
	const [dialogSize, setDialogSize] = useState<Dimensions>({height: '', width: ''});
	const [onDialogClosed, setOnDialogClosed] = useState<(() => void) | null>(null);

	const openDialog = (title: string | null, content: ReactNode, dimensions: Dimensions, dialogClosed = null) => {
		if (title) setDialogTitle(title);
		setDialogContent(content);
		setDialogSize({height: dimensions.height, width: dimensions.width});
		if (dialogClosed) setOnDialogClosed(dialogClosed);
	};

	const closeDialog = (e?: React.MouseEvent) => {
		e?.stopPropagation();
		setDialogContent(null);
		setDialogTitle(null);
		onDialogClosed?.();
		setOnDialogClosed(null);
	};

	const DialogComponent = () => (
		<Dialog open={!!dialogContent}>
			<DialogPortal>
				<DialogContent data-testid='dialog' aria-hidden={false} style={{maxHeight: dialogSize.height, width: dialogSize.width}} className={'[&>button]:hidden z-[99] !pointer-events-auto p-0 h-[80%] font-inter bg-white block max-w-[95%]'}>
					<div className='bg-white h-full rounded-[12px] flex flex-col' aria-hidden={false}>
						<DialogHeader className={clsx(!dialogTitle && 'hidden')}>
							<DialogTitle>
								<div className='mb-[12px] mt-[24px] w-full flex justify-between items-center ps-[30px] pe-[20px]'>
									<DialogDescription className='text-[20px] font-semibold'>{dialogTitle}</DialogDescription>
									<img role='button' tabIndex={0} onKeyDown={spaceClick} onClick={closeDialog} className='cursor-pointer rounded-full' src='icons/close.svg' alt='close' width={24} height={24} />
								</div>
							</DialogTitle>
						</DialogHeader>
						<div className='overflow-auto flex-1'>{dialogContent}</div>
					</div>
				</DialogContent>
			</DialogPortal>
		</Dialog>
	);

	return {openDialog, DialogComponent, closeDialog};
};
