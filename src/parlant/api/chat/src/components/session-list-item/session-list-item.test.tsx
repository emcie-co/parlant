import {cleanup, fireEvent, MatcherOptions, render} from '@testing-library/react';
import {describe, expect, it, vi} from 'vitest';
import {Matcher} from 'vite';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom/vitest';

import {SessionInterface} from '@/utils/interfaces';
import Session, {DeleteDialog} from './session-list-item';

const session: SessionInterface | null = {id: 'session1', title: 'Session One', customer_id: '', agent_id: '', creation_utc: new Date().toLocaleString()};

vi.mock('@/utils/api', () => ({
	deleteData: vi.fn(() => Promise.resolve()),
}));

const setSessionFn = vi.fn();
const openDialogFn = vi.fn();
vi.mock('react', async () => {
	const actualReact = await vi.importActual('react');
	return {
		...actualReact,
		useContext: vi.fn(() => ({
			setSessionId: setSessionFn,
			setAgentId: vi.fn(),
			setSessions: vi.fn(),
			openDialog: openDialogFn,
		})),
	};
});

describe(Session, () => {
	let getByTestId: (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
	let rerender: (ui: React.ReactNode) => void;
	let container: HTMLElement;

	const openDeleteDialog = async (): Promise<void> => {
		const moreBtn = getByTestId('menu-button');
		await userEvent.click(moreBtn);
		const deleteBtn = getByTestId('delete');
		await fireEvent.click(deleteBtn);
	};

	const deleteSession = async (): Promise<{dialog: HTMLElement}> => {
		const dialog = getByTestId('deleteDialogContent');
		const dialogDeleteButton = getByTestId('gradient-button');
		fireEvent.click(dialogDeleteButton);
		return {dialog};
	};

	beforeEach(() => {
		const utils = render(<Session editingTitle={null} setEditingTitle={vi.fn()} session={session as SessionInterface} refetch={vi.fn()} isSelected={true} />);
		getByTestId = utils.getByTestId as (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
		rerender = utils.rerender;
		container = utils.container;

		vi.clearAllMocks();
	});

	afterEach(() => cleanup());

	it('component should be rendered', () => {
		const div = getByTestId('session');
		expect(div).toBeInTheDocument();
	});

	it('delete button should open a delete dialog', async () => {
		await openDeleteDialog();
		expect(openDialogFn).toHaveBeenCalled();
	});

	it("dialog's delete button should fire a delete event", async () => {
		const deleteClickedFn = vi.fn();
		rerender(<DeleteDialog closeDialog={vi.fn()} deleteClicked={deleteClickedFn} session={session} />);
		await deleteSession();
		expect(deleteClickedFn).toBeCalled();
	});

	it('text field opened when editing the current session', async () => {
		rerender(<Session editingTitle={session.id} setEditingTitle={vi.fn()} session={session as SessionInterface} refetch={vi.fn()} isSelected={false} />);
		const textfields = container.querySelector('input');
		expect(textfields).toBeInTheDocument();
	});

	it('text field closed when "cancel edit" button is clicked', async () => {
		const setEditingTitleFn = vi.fn();
		rerender(<Session editingTitle={session.id} setEditingTitle={setEditingTitleFn} session={session as SessionInterface} refetch={vi.fn()} isSelected={false} />);
		const cancelBtn = getByTestId('cancel');
		fireEvent.click(cancelBtn);
		expect(setEditingTitleFn).toBeCalledWith(null);
	});

	it('session selection should be enabled when not editing another session', async () => {
		rerender(<Session editingTitle={null} setEditingTitle={vi.fn()} session={session as SessionInterface} refetch={vi.fn()} isSelected={false} />);
		const currSession = getByTestId('session');
		fireEvent.click(currSession);
		expect(setSessionFn).toBeCalledWith(session.id);
	});

	it('session selection should be disabled when editing another session', async () => {
		rerender(<Session editingTitle='session2' setEditingTitle={vi.fn()} session={session as SessionInterface} refetch={vi.fn()} isSelected={false} />);
		const currSession = getByTestId('session');
		fireEvent.click(currSession);
		expect(setSessionFn).not.toBeCalled();
	});
});
