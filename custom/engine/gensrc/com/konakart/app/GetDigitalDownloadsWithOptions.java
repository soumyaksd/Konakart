package com.konakart.app;

import com.konakart.appif.*;

/**
 *  The KonaKart Custom Engine - GetDigitalDownloadsWithOptions - Generated by CreateKKCustomEng
 */
@SuppressWarnings("all")
public class GetDigitalDownloadsWithOptions
{
    KKEng kkEng = null;

    /**
     * Constructor
     */
     public GetDigitalDownloadsWithOptions(KKEng _kkEng)
     {
         kkEng = _kkEng;
     }

     public DigitalDownloadIf[] getDigitalDownloadsWithOptions(String sessionId, int languageId, FetchDigitalDownloadOptionsIf ddOptions, FetchProductOptionsIf prodOptions) throws KKException
     {
         return kkEng.getDigitalDownloadsWithOptions(sessionId, languageId, ddOptions, prodOptions);
     }
}
