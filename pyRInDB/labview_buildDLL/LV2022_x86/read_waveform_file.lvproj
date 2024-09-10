<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="22308000">
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">true</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="read_all_waveform_data.vi" Type="VI" URL="../read_all_waveform_data.vi"/>
		<Item Name="read_all_waveform_length.vi" Type="VI" URL="../read_all_waveform_length.vi"/>
		<Item Name="read_waveform_data.vi" Type="VI" URL="../read_waveform_data.vi"/>
		<Item Name="read_waveform_length.vi" Type="VI" URL="../read_waveform_length.vi"/>
		<Item Name="read_waveform_number.vi" Type="VI" URL="../read_waveform_number.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="Close DWDT Array Dlog File+.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Close DWDT Array Dlog File+.vi"/>
				<Item Name="Close WDT Array Dlog File+.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Close WDT Array Dlog File+.vi"/>
				<Item Name="compatOpenFileOperation.vi" Type="VI" URL="/&lt;vilib&gt;/_oldvers/_oldvers.llb/compatOpenFileOperation.vi"/>
				<Item Name="compatOverwrite.vi" Type="VI" URL="/&lt;vilib&gt;/_oldvers/_oldvers.llb/compatOverwrite.vi"/>
				<Item Name="Find First Error.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find First Error.vi"/>
				<Item Name="Open Create Replace DWDT Array Dlog File.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Open Create Replace DWDT Array Dlog File.vi"/>
				<Item Name="Open Create Replace WDT Array Dlog File.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Open Create Replace WDT Array Dlog File.vi"/>
				<Item Name="Read DWDT Array Dlog File+.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Read DWDT Array Dlog File+.vi"/>
				<Item Name="Read Waveform from File (Digital).vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Read Waveform from File (Digital).vi"/>
				<Item Name="Read Waveform from File.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Read Waveform from File.vi"/>
				<Item Name="Read Waveforms from File.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Read Waveforms from File.vi"/>
				<Item Name="Read WDT Array Dlog File+.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTFileIO.llb/Read WDT Array Dlog File+.vi"/>
			</Item>
		</Item>
		<Item Name="Build Specifications" Type="Build">
			<Item Name="read_waveform_file" Type="DLL">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{BB1FAF11-DF59-49A2-A5A7-05C921E2B0E7}</Property>
				<Property Name="App_INI_GUID" Type="Str">{06231C3A-C922-4E54-92C7-625C4280CC4A}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="App_serverType" Type="Int">0</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{29A84AE7-CED9-40A5-AB54-1335FD5C2086}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">read_waveform_file</Property>
				<Property Name="Bld_excludeInlineSubVIs" Type="Bool">true</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../builds</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{B9A594AE-5F67-4BC5-9E15-B3E8A302968F}</Property>
				<Property Name="Bld_version.build" Type="Int">8</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">wvfRead.dll</Property>
				<Property Name="Destination[0].path" Type="Path">../builds/wvfRead.dll</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../builds/data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Dll_compatibilityWith2011" Type="Bool">false</Property>
				<Property Name="Dll_delayOSMsg" Type="Bool">true</Property>
				<Property Name="Dll_headerGUID" Type="Str">{BF32247C-A35F-4E65-BE4F-736827719178}</Property>
				<Property Name="Dll_libGUID" Type="Str">{3DB688F2-4D5D-4BCC-8C9B-3E30570B646F}</Property>
				<Property Name="Dll_privateExecSys" Type="Bool">true</Property>
				<Property Name="Source[0].itemID" Type="Str">{9BAC7821-E2C0-42B4-85CD-D1EA7AB939FB}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoName" Type="Str">return value</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoOutputIdx" Type="Int">1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[0]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoInputIdx" Type="Int">3</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoName" Type="Str">path</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[1]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]CallingConv" Type="Int">0</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]Name" Type="Str">read_waveform_length</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoInputIdx" Type="Int">2</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoName" Type="Str">N</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfo[2]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfoCPTM" Type="Bin">)D#!!!!!!!5!"!!!!!V!!Q!'&lt;'6O:X2I!!!(1!=!!5Y!$E!Q`````Q2Q982I!!!E!0!!"!!!!!%!!A!$!Q!!+!!!!!!!!!E!!!!)!!!#%!!!!!!"!!1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfoDescription" Type="Bin">5G6U&gt;8*O)(2I:3"M:7ZH&gt;'AM)'FO)(.B&lt;8"M:8-M)'^G)(2I:3"/&gt;'AA&gt;W&amp;W:7:P=GUA;7YA93"-97*W;76X)(&gt;B&gt;G6G&lt;X*N)':J&lt;'5</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfoisCustomDesc" Type="Int">1</Property>
				<Property Name="Source[1].ExportedVI.VIProtoInfoVIProtoItemCount" Type="Int">3</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/read_waveform_length.vi</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">ExportedVI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoName" Type="Str">return value</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[0]VIProtoPassBy" Type="Int">0</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoInputIdx" Type="Int">3</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoName" Type="Str">path</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[1]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoInputIdx" Type="Int">2</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoName" Type="Str">N</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[2]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoLenInput" Type="Int">4</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoName" Type="Str">data</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoOutputIdx" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[3]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoDir" Type="Int">3</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoName" Type="Str">dataLen</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[4]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]CallingConv" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]Name" Type="Str">read_waveform_data</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoName" Type="Str">dt</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoOutputIdx" Type="Int">0</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfo[5]VIProtoPassBy" Type="Int">0</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfoCPTM" Type="Bin">)D#!!!!!!!9!#5!+!!*E&gt;!!!$5!+!!&gt;/&gt;7VF=GFD!"*!1!!"`````Q!""'2B&gt;'%!!!&gt;!"Q!"4A!/1$$`````"("B&gt;'A!!#1!]!!%!!!!!A!$!!1$!!!I!!!*!!!!#1!!!!A!!!))!!!!!!%!"1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfoDescription" Type="Bin">5G6B:#"E982B)':S&lt;WUA&gt;'BF)%ZU;#"X98:F:G^S&lt;3"J&lt;C"B)%RB9H:J:8=A&gt;W&amp;W:7:P=GUA:GFM:1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfoisCustomDesc" Type="Int">1</Property>
				<Property Name="Source[2].ExportedVI.VIProtoInfoVIProtoItemCount" Type="Int">6</Property>
				<Property Name="Source[2].itemID" Type="Ref">/My Computer/read_waveform_data.vi</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[2].type" Type="Str">ExportedVI</Property>
				<Property Name="Source[3].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoName" Type="Str">return value</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[0]VIProtoPassBy" Type="Int">0</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoInputIdx" Type="Int">3</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoName" Type="Str">path</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[1]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoInputIdx" Type="Int">2</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoName" Type="Str">N</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[2]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoLenInput" Type="Int">2</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoName" Type="Str">dt</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoOutputIdx" Type="Int">0</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[3]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoLenInput" Type="Int">5</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoName" Type="Str">data</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoOutputIdx" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[4]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]CallingConv" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]Name" Type="Str">read_all_waveform_data</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoDir" Type="Int">3</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoName" Type="Str">lenData</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfo[5]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfoCPTM" Type="Bin">)D#!!!!!!!9!"1!+!!!11%!!!@````]!!!*E&gt;!!!%E"!!!(`````!!!%:'&amp;U91!!"U!$!!&amp;/!!Z!-0````]%='&amp;U;!!!*!$Q!!1!!1!#!!-!"!-!!#A!!!E!!!!*!!!!#A!!!B!!!!!!!1!&amp;</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfoDescription" Type="Bin">5G6B:#"E982B)':S&lt;WUA&gt;'BF)':J=H.U)%YA&gt;W&amp;W:7:P=GVT)'FO)'%A4'&amp;C&gt;GFF&gt;S"X98:F:G^S&lt;3"G;7RF</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfoisCustomDesc" Type="Int">1</Property>
				<Property Name="Source[3].ExportedVI.VIProtoInfoVIProtoItemCount" Type="Int">6</Property>
				<Property Name="Source[3].itemID" Type="Ref">/My Computer/read_all_waveform_data.vi</Property>
				<Property Name="Source[3].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[3].type" Type="Str">ExportedVI</Property>
				<Property Name="Source[4].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoName" Type="Str">return value</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoOutputIdx" Type="Int">0</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[0]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]CallingConv" Type="Int">1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]Name" Type="Str">read_waveform_number</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoInputIdx" Type="Int">1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoName" Type="Str">path</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfo[1]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfoCPTM" Type="Bin">)D#!!!!!!!-!$5!$!!:M:7ZH&gt;'A!!!Z!-0````]%='&amp;U;!!!'!$Q!!)!!!!"!Q!!#!!!#1!!!B!!!!!!!1!#</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfoDescription" Type="Bin">5G6B:#"U;'5A&lt;H6N9G6S)'^G)(&gt;B&gt;G6G&lt;X*N=S"J&lt;C"B)%RB9H:J:8=A&gt;W&amp;W:7:P=GUA:GFM:1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfoisCustomDesc" Type="Int">1</Property>
				<Property Name="Source[4].ExportedVI.VIProtoInfoVIProtoItemCount" Type="Int">2</Property>
				<Property Name="Source[4].itemID" Type="Ref">/My Computer/read_waveform_number.vi</Property>
				<Property Name="Source[4].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[4].type" Type="Str">ExportedVI</Property>
				<Property Name="Source[5].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoName" Type="Str">return value</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[0]VIProtoPassBy" Type="Int">0</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoInputIdx" Type="Int">2</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoName" Type="Str">path</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[1]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoDir" Type="Int">0</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoInputIdx" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoLenInput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoName" Type="Str">N</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoOutputIdx" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[2]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]CallingConv" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]Name" Type="Str">read_all_waveform_length</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoDir" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoInputIdx" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoLenInput" Type="Int">2</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoLenOutput" Type="Int">-1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoName" Type="Str">length</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoOutputIdx" Type="Int">0</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfo[3]VIProtoPassBy" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfoCPTM" Type="Bin">)D#!!!!!!!5!"1!$!!!51%!!!@````]!!!:M:7ZH&gt;'A!!!&gt;!!Q!"4A!/1$$`````"("B&gt;'A!!"Y!]!!$!!%!!A!$!Q!!%!!!#1!!!!I!!!)1!!!!!!%!"!</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfoDescription" Type="Bin">5G6B:#"U;'5A&lt;'6O:X2I)'^G)(2I:3"G;8*T&gt;#"/)(&gt;B&gt;G6G&lt;X*N=S"J&lt;C"B)%RB9H:J:8=A&gt;W&amp;W:7:P=GUA:GFM:1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfoisCustomDesc" Type="Int">1</Property>
				<Property Name="Source[5].ExportedVI.VIProtoInfoVIProtoItemCount" Type="Int">4</Property>
				<Property Name="Source[5].itemID" Type="Ref">/My Computer/read_all_waveform_length.vi</Property>
				<Property Name="Source[5].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[5].type" Type="Str">ExportedVI</Property>
				<Property Name="SourceCount" Type="Int">6</Property>
				<Property Name="TgtF_fileDescription" Type="Str">read_waveform_file</Property>
				<Property Name="TgtF_internalName" Type="Str">read_waveform_file</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2023 </Property>
				<Property Name="TgtF_productName" Type="Str">read_waveform_file</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{D50EB025-88B9-4D5C-83D2-29EB97092DA2}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">wvfRead.dll</Property>
				<Property Name="TgtF_versionIndependent" Type="Bool">true</Property>
			</Item>
		</Item>
	</Item>
</Project>
