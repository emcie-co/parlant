// import { describe, expect, it, vi } from 'vitest';
// import { fireEvent, render, screen } from '@testing-library/react';

// import { EventInterface, ServerStatus } from '@/utils/interfaces';
// import Message from './message';

// const serverStatuses: ServerStatus[] = ['pending', 'error', 'accepted', 'acknowledged', 'processing', 'typing', 'ready'];

// const event: EventInterface = {
//     correlation_id: '',
//     creation_utc: new Date(),
//     data: {message: 'Hi'},
//     kind: 'message',
//     offset: 0,
//     serverStatus: 'pending',
//     source: 'customer'
// };

// describe(Message, () => {
//     let getByTestId: typeof screen.getByTestId;
//     let queryByTestId: typeof screen.queryByTestId;
//     let rerender: (ui: React.ReactNode) => void;

//     beforeEach(() => {
//         const utils = render(<Message isContinual={false} event={event}/>);
//         getByTestId = utils.getByTestId as typeof getByTestId;
//         queryByTestId = utils.queryByTestId as typeof queryByTestId;;
//         rerender = utils.rerender;
//     });

//     it('component should be rendered', () => {
//         const message = getByTestId('message');
//         expect(message).toBeInTheDocument();
//     });

//     it('message has the valid icon', () => {
//         for (const serverStatus of serverStatuses) {
//             rerender(<Message isContinual={false} event={{...event, serverStatus}}/>);
//             const icon = getByTestId(serverStatus);
//             expect(icon).toBeInTheDocument();
//         }
//     });

//     it('client messages should not have a regenerate button', () => {
//         rerender(<Message isContinual={false} event={event}/>);
//         const button = queryByTestId('regenerate-button');
//         expect(button).toBeNull();
//     });

//     it('server messages have a regenerate button', () => {
//         rerender(<Message isContinual={false} event={{...event, source: 'ai_agent'}}/>);
//         const button = queryByTestId('regenerate-button');
//         expect(button).toBeInTheDocument();
//     });

//     it('clicking regenrate should call the regenrate message function', () => {
//         const regenerateMessageFn = vi.fn();
//         rerender(<Message isContinual={false} event={{...event, source: 'ai_agent'}} regenerateMessageFn={regenerateMessageFn}/>);
//         const button = getByTestId('regenerate-button');
//         fireEvent.click(button);
//         expect(regenerateMessageFn).toHaveBeenCalled();
//     });
// });
