export function groupBy<T>(array: T[], keyFn: (item: T) => string | number): Record<string, T[]> {
    return array.reduce((result: Record<string, T[]>, item: T) => {
      let key = keyFn(item);
      if (!key) key = key?.toString();
      if (!result[key]) {
        result[key] = [];
      }
      result[key].push(item);
      return result;
    }, {});
  }