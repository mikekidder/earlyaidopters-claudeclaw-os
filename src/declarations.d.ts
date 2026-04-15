declare module 'better-sqlite3' {
  namespace Database {
    interface Database {
      [key: string]: any;
    }
  }
  interface DatabaseConstructor {
    new (filename: string, options?: any): Database.Database;
    (filename: string, options?: any): Database.Database;
  }
  const Database: DatabaseConstructor;
  export = Database;
}
declare module 'js-yaml';
declare module 'qrcode-terminal';
