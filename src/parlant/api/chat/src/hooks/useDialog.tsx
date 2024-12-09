import { useState, ReactNode } from 'react';
import { Dialog, DialogContent, DialogHeader } from '@/components/ui/dialog';
import { DialogDescription, DialogTitle } from '@radix-ui/react-dialog';
import { spaceClick } from '@/utils/methods';
import clsx from 'clsx';

interface UseDialogReturn {
  openDialog: (title: string, content: ReactNode, dimensions: Dimensions) => void;
  DialogComponent: () => JSX.Element;
  closeDialog: (e?: React.MouseEvent) => void;
};

export interface Dimensions {
  height: string;
  width: string;
};

export const useDialog = (): UseDialogReturn => {
  const [dialogTitle, setDialogTitle] = useState<ReactNode>(null);
  const [dialogContent, setDialogContent] = useState<ReactNode>(null);
  const [dialogSize, setDialogSize] = useState<Dimensions>({height: '', width: ''});
  const [onDialogClosed, setOnDialogClosed] = useState<(() => void) | null>(null);

  const openDialog = (title: string, content: ReactNode, dimensions: Dimensions, dialogClosed = null) => {
      setDialogTitle(title);
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
        <DialogContent data-testid="dialog" style={{maxHeight: dialogSize.height, width: dialogSize.width}} className={'[&>button]:hidden p-0 h-[80%] font-ubuntu-sans bg-white block max-w-[95%]'}>
                <div className='bg-white h-full rounded-[12px] flex flex-col'>
                    <DialogHeader className={clsx(!dialogTitle && 'hidden')}>
                        <DialogTitle>
                            <div className='h-[68px] w-full flex justify-between items-center ps-[30px] pe-[20px] border-b-[#EBECF0] border-b-[0.6px]'>
                                <DialogDescription className='text-[16px] font-bold'>{dialogTitle}</DialogDescription>
                                <img role='button' tabIndex={0} onKeyDown={spaceClick} onClick={closeDialog} className='cursor-pointer rounded-full hover:bg-[#F5F6F8] p-[10px]' src="icons/close.svg" alt="close" height={30} width={30}/>
                            </div>
                        </DialogTitle>
                    </DialogHeader>
                    <div className='overflow-auto flex-1'>
                      {dialogContent}
                    </div>
                </div>
            </DialogContent>
    </Dialog>
  );

  return {openDialog, DialogComponent, closeDialog};
};
