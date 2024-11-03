import { fireEvent, render, RenderResult } from '@testing-library/react';
import { expect, it } from 'vitest';
import '@testing-library/jest-dom/vitest';

import { useDialog } from './useDialog';

const TestComponent = () => {
    const { DialogComponent, openDialog, closeDialog } = useDialog();
    
    return (
      <div>
        <button onClick={() => openDialog('title', <div>Mocked-Dialog</div>, {height: '0px', width: '0px'})}>Open Dialog</button>
        <button onClick={() => closeDialog()}>Close Dialog</button>
        <DialogComponent />
      </div>
    );
  };

const openDialog = (): RenderResult => {
    const utils = render(<TestComponent />);
    const button = utils.getByText('Open Dialog');
    fireEvent.click(button);
    return utils;
};

describe(useDialog, () => {
    it('openDialog function should open the dialog', async () => {
        const {getByText} = openDialog();
        const dialog = getByText('Mocked-Dialog');
        expect(dialog).toBeInTheDocument();
    });

    it('closeDialog function should close the dialog', async () => {
        const {getByText} = openDialog();
        const button = getByText('Close Dialog');
        const dialog = getByText('Mocked-Dialog');
        fireEvent.click(button);
        expect(dialog).not.toBeInTheDocument();
    });
});