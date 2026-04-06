export const getDateStr = (date: Date | string): string => {
    date = new Date(date);
    const options: Intl.DateTimeFormatOptions = { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    };

    return date.toLocaleDateString('en-US', options);
};

export const getTimeStr = (date: Date |string): string => {
    date = new Date(date);
    const options: Intl.DateTimeFormatOptions = {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    };
  
    return date.toLocaleTimeString('en-US', options);
  };